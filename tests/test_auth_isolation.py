"""Multi-User Authentication and Data Isolation Test Suite for SecondSelf.

Validates the 9 core requirements from the specification:
1. User A can register and log in.
2. User B can independently register and log in.
3. User A cannot see User B's notes.
4. User B cannot see User A's knowledge graph.
5. Editing or deleting a note only affects its owner.
6. Logout prevents access to protected data.
7. Direct access without authentication is blocked.
8. Admin access is restricted to authorized administrators.
9. Existing records are preserved and assigned to the default owner.
10. Grounded RAG query isolation (User B queries never cite or retrieve User A notes).
11. Multi-tenant capture store isolation.
12. Password reset lifecycle and session security.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure secondself project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app
import ask
import build_graph
import capture
import classify
from lib import auth
import lib.database as db
from lib.database import DEFAULT_USER_ID
import lib.embeddings as emb
from lib.models import CaptureResult, GraphEdge, GraphNode, User, WikiNote
import lib.startup as startup
import lib.storage as storage
import link
import pipeline


class TestMultiUserDataIsolation(unittest.TestCase):
    """Rigorous multi-tenant isolation and security test suite."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name).resolve()

        self._old_env = {
            "SECONDSELF_ROOT": os.environ.get("SECONDSELF_ROOT"),
            "SECONDSELF_DATA_DIR": os.environ.get("SECONDSELF_DATA_DIR"),
            "SECONDSELF_DB_PATH": os.environ.get("SECONDSELF_DB_PATH"),
            "SECONDSELF_AUTO_LOGIN_ADMIN": os.environ.get("SECONDSELF_AUTO_LOGIN_ADMIN"),
        }
        os.environ["SECONDSELF_ROOT"] = str(self.test_root)
        os.environ["SECONDSELF_DATA_DIR"] = str(self.test_root / "data")
        os.environ["SECONDSELF_DB_PATH"] = str(self.test_root / "data" / "secondself.db")
        os.environ.pop("SECONDSELF_AUTO_LOGIN_ADMIN", None)

        storage.set_project_root(self.test_root)
        storage.get_raw_dir(self.test_root).mkdir(parents=True, exist_ok=True)
        storage.get_data_dir(self.test_root).mkdir(parents=True, exist_ok=True)
        wiki_dir = storage.get_wiki_dir(self.test_root)
        wiki_dir.mkdir(parents=True, exist_ok=True)
        for cat in ("Projects", "Areas", "Resources", "Archives"):
            (wiki_dir / cat).mkdir(parents=True, exist_ok=True)

        # Clear startup & Streamlit caches
        startup._STARTUP_COMPLETED = False
        app.load_graph.clear()
        app.load_embeddings.clear()
        app.load_note_embeddings.clear()

        # Initialize fresh multi-tenant database
        db.init_db(self.test_root)

    def tearDown(self) -> None:
        storage.set_project_root(None)
        startup._STARTUP_COMPLETED = False
        app.load_graph.clear()
        app.load_embeddings.clear()
        app.load_note_embeddings.clear()

        for k, v in self._old_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Requirement 1 & 2: Independent User Registration and Login
    # -------------------------------------------------------------------------

    def test_01_user_registration_and_login_success(self) -> None:
        """Requirement 1: User A can register with email and password and log in successfully."""
        ok, msg, user_a = auth.register_user(
            email="alice@company.org",
            password="SecurePassword123!",
            full_name="Alice Smith",
            root=self.test_root,
        )
        self.assertTrue(ok, f"Registration failed: {msg}")
        self.assertIsNotNone(user_a)
        self.assertEqual(user_a.email, "alice@company.org")
        self.assertEqual(user_a.full_name, "Alice Smith")
        self.assertEqual(user_a.role, "user")
        self.assertTrue(user_a.is_active)

        # Log in with valid credentials
        login_ok, login_msg, session, logged_in_user = auth.login_user(
            email="alice@company.org",
            password="SecurePassword123!",
            root=self.test_root,
        )
        self.assertTrue(login_ok, f"Login failed: {login_msg}")
        self.assertIsNotNone(session)
        self.assertIsNotNone(logged_in_user)
        self.assertEqual(logged_in_user.id, user_a.id)

        # Verify session token resolves authenticated user
        auth_user = auth.authenticate_session(session.token, root=self.test_root)
        self.assertIsNotNone(auth_user)
        self.assertEqual(auth_user.id, user_a.id)

    def test_02_independent_user_b_registration_and_login(self) -> None:
        """Requirement 2: User B can independently register and log in."""
        ok_a, _, user_a = auth.register_user(
            email="alice@company.org",
            password="AlicePassword123!",
            full_name="Alice Smith",
            root=self.test_root,
        )
        self.assertTrue(ok_a)

        ok_b, msg_b, user_b = auth.register_user(
            email="bob@startup.io",
            password="BobPassword456!",
            full_name="Bob Jones",
            root=self.test_root,
        )
        self.assertTrue(ok_b, f"Bob registration failed: {msg_b}")
        self.assertIsNotNone(user_b)
        self.assertNotEqual(user_a.id, user_b.id)

        # Login User B
        login_ok, _, session_b, logged_in_b = auth.login_user(
            email="bob@startup.io",
            password="BobPassword456!",
            root=self.test_root,
        )
        self.assertTrue(login_ok)
        self.assertEqual(logged_in_b.id, user_b.id)

        # Invalid password rejected
        bad_ok, bad_msg, bad_sess, _ = auth.login_user(
            email="bob@startup.io",
            password="WrongPassword!",
            root=self.test_root,
        )
        self.assertFalse(bad_ok)
        self.assertIsNone(bad_sess)
        self.assertIn("Invalid", bad_msg)

    # -------------------------------------------------------------------------
    # Requirement 3: Note Isolation
    # -------------------------------------------------------------------------

    def test_03_user_a_cannot_see_user_b_notes(self) -> None:
        """Requirement 3: Notes created by User A are strictly invisible to User B."""
        _, _, user_a = auth.register_user("alice@domain.com", "Pass12345!", root=self.test_root)
        _, _, user_b = auth.register_user("bob@domain.com", "Pass12345!", root=self.test_root)

        # User A creates a confidential note
        note_a = WikiNote(
            id="alice_secret_01",
            raw_id="raw_alice_01",
            para="Projects",
            tags=["confidential", "q3-roadmap"],
            summary="Confidential Strategy for Alice",
            created="2026-11-13T10:00:00Z",
            links=[],
            body="Project Titan is the internal code name for Alice's secret product.",
            user_id=user_a.id,
        )
        storage.write_wiki_note(note_a, root=self.test_root)

        # User B creates a different note
        note_b = WikiNote(
            id="bob_note_01",
            raw_id="raw_bob_01",
            para="Resources",
            tags=["recipes"],
            summary="Bob Sourdough Bread Recipe",
            created="2026-11-13T10:05:00Z",
            links=[],
            body="Flour, water, salt, sourdough starter fermentation.",
            user_id=user_b.id,
        )
        storage.write_wiki_note(note_b, root=self.test_root)

        # 1. Read notes scoped to User A
        notes_for_a = storage.read_wiki_notes(root=self.test_root, user_id=user_a.id)
        note_ids_a = {n.id for n in notes_for_a}
        self.assertIn("alice_secret_01", note_ids_a)
        self.assertNotIn("bob_note_01", note_ids_a)

        # 2. Read notes scoped to User B
        notes_for_b = storage.read_wiki_notes(root=self.test_root, user_id=user_b.id)
        note_ids_b = {n.id for n in notes_for_b}
        self.assertIn("bob_note_01", note_ids_b)
        self.assertNotIn("alice_secret_01", note_ids_b)

        # 3. Database direct query isolation
        db_notes_a = db.get_all_notes(user_id=user_a.id, root=self.test_root)
        self.assertEqual(len(db_notes_a), 1)
        self.assertEqual(db_notes_a[0]["id"], "alice_secret_01")

        db_notes_b = db.get_all_notes(user_id=user_b.id, root=self.test_root)
        self.assertEqual(len(db_notes_b), 1)
        self.assertEqual(db_notes_b[0]["id"], "bob_note_01")

        # 4. User B cannot fetch User A's note by ID
        target_a_from_b = db.get_note_by_id("alice_secret_01", user_id=user_b.id, root=self.test_root)
        self.assertIsNone(target_a_from_b)

    # -------------------------------------------------------------------------
    # Requirement 4: Knowledge Graph Isolation
    # -------------------------------------------------------------------------

    def test_04_user_b_cannot_see_user_a_knowledge_graph(self) -> None:
        """Requirement 4: Knowledge graph nodes and edges are strictly isolated per user."""
        _, _, user_a = auth.register_user("alice@domain.com", "Pass12345!", root=self.test_root)
        _, _, user_b = auth.register_user("bob@domain.com", "Pass12345!", root=self.test_root)

        # Build graph for User A
        nodes_a = [
            GraphNode(id="a_node1", label="Alice Alpha", para="Projects", tags=["alpha"], summary="S1", content_preview="P1", group="Projects", user_id=user_a.id),
            GraphNode(id="a_node2", label="Alice Beta", para="Projects", tags=["beta"], summary="S2", content_preview="P2", group="Projects", user_id=user_a.id),
        ]
        edges_a = [
            GraphEdge(source="a_node1", target="a_node2", weight=0.9, type="semantic", user_id=user_a.id)
        ]
        db.save_graph_records(nodes_a, edges_a, {"node_count": 2, "edge_count": 1}, user_id=user_a.id, root=self.test_root)

        # Build graph for User B
        nodes_b = [
            GraphNode(id="b_node1", label="Bob Omega", para="Areas", tags=["omega"], summary="S3", content_preview="P3", group="Areas", user_id=user_b.id),
        ]
        edges_b = []
        db.save_graph_records(nodes_b, edges_b, {"node_count": 1, "edge_count": 0}, user_id=user_b.id, root=self.test_root)

        # Verify User A's graph from DB
        a_graph_nodes = db.get_graph_nodes(user_id=user_a.id, root=self.test_root)
        a_graph_edges = db.get_graph_edges(user_id=user_a.id, root=self.test_root)
        self.assertEqual(len(a_graph_nodes), 2)
        self.assertEqual(len(a_graph_edges), 1)
        self.assertEqual({n["id"] for n in a_graph_nodes}, {"a_node1", "a_node2"})

        # Verify User B cannot see User A's graph
        b_graph_nodes = db.get_graph_nodes(user_id=user_b.id, root=self.test_root)
        b_graph_edges = db.get_graph_edges(user_id=user_b.id, root=self.test_root)
        self.assertEqual(len(b_graph_nodes), 1)
        self.assertEqual(len(b_graph_edges), 0)
        self.assertEqual(b_graph_nodes[0]["id"], "b_node1")

        # Verify stats summary isolation
        stats_a = db.get_stats_summary(user_id=user_a.id, root=self.test_root)
        self.assertEqual(stats_a["graph_nodes"], 2)
        self.assertEqual(stats_a["graph_edges"], 1)

        stats_b = db.get_stats_summary(user_id=user_b.id, root=self.test_root)
        self.assertEqual(stats_b["graph_nodes"], 1)
        self.assertEqual(stats_b["graph_edges"], 0)

    # -------------------------------------------------------------------------
    # Requirement 5: Scoped Note Editing and Deletion
    # -------------------------------------------------------------------------

    def test_05_editing_or_deleting_only_affects_owner(self) -> None:
        """Requirement 5: User B cannot delete or modify User A's note."""
        _, _, user_a = auth.register_user("alice@domain.com", "Pass12345!", root=self.test_root)
        _, _, user_b = auth.register_user("bob@domain.com", "Pass12345!", root=self.test_root)

        note_a = WikiNote(
            id="alice_note_42",
            raw_id="raw_42",
            para="Projects",
            tags=["critical"],
            summary="Alice Mission Critical Note",
            created="2026-11-13T10:00:00Z",
            links=[],
            body="Initial body content.",
            user_id=user_a.id,
        )
        storage.write_wiki_note(note_a, root=self.test_root)

        # User B attempts to delete User A's note -> must fail and return False
        b_del_result = storage.delete_wiki_note("alice_note_42", user_id=user_b.id, root=self.test_root)
        self.assertFalse(b_del_result)

        # Verify note still exists intact in SQLite and disk
        note_still_there = db.get_note_by_id("alice_note_42", user_id=user_a.id, root=self.test_root)
        self.assertIsNotNone(note_still_there)
        self.assertEqual(note_still_there["summary"], "Alice Mission Critical Note")

        # User A successfully edits their own note
        updated_note_a = WikiNote(
            id="alice_note_42",
            raw_id="raw_42",
            para="Projects",
            tags=["critical", "updated"],
            summary="Alice Updated Note",
            created="2026-11-13T10:00:00Z",
            links=[],
            body="Updated content by Alice.",
            user_id=user_a.id,
        )
        storage.write_wiki_note(updated_note_a, root=self.test_root)
        saved = db.get_note_by_id("alice_note_42", user_id=user_a.id, root=self.test_root)
        self.assertEqual(saved["summary"], "Alice Updated Note")

        # User A deletes their own note -> succeeds
        a_del_result = storage.delete_wiki_note("alice_note_42", user_id=user_a.id, root=self.test_root)
        self.assertTrue(a_del_result)
        self.assertIsNone(db.get_note_by_id("alice_note_42", user_id=user_a.id, root=self.test_root))

    # -------------------------------------------------------------------------
    # Requirement 6: Logout Invalidation
    # -------------------------------------------------------------------------

    def test_06_logout_prevents_access_to_protected_data(self) -> None:
        """Requirement 6: Logging out invalidates the session token immediately."""
        _, _, user = auth.register_user("logout_test@company.com", "Pass12345!", root=self.test_root)
        _, _, session, _ = auth.login_user("logout_test@company.com", "Pass12345!", root=self.test_root)

        self.assertIsNotNone(auth.authenticate_session(session.token, root=self.test_root))

        # Perform logout
        logout_ok = auth.logout_user(session.token, root=self.test_root)
        self.assertTrue(logout_ok)

        # Subsequent authentication with invalidated token must fail
        post_logout_user = auth.authenticate_session(session.token, root=self.test_root)
        self.assertIsNone(post_logout_user)

    # -------------------------------------------------------------------------
    # Requirement 7: Block Unauthenticated Direct Access
    # -------------------------------------------------------------------------

    def test_07_unauthenticated_access_is_blocked(self) -> None:
        """Requirement 7: Random or expired session tokens cannot access protected sessions."""
        fake_token = "invalid_token_99999999999999999999"
        self.assertIsNone(auth.authenticate_session(fake_token, root=self.test_root))
        self.assertIsNone(auth.authenticate_session("", root=self.test_root))
        self.assertIsNone(auth.authenticate_session(None, root=self.test_root))

    # -------------------------------------------------------------------------
    # Requirement 8: Server-Side Role Enforcement (Admin Portal Security)
    # -------------------------------------------------------------------------

    def test_08_admin_access_restricted_to_authorized_administrators(self) -> None:
        """Requirement 8: Standard users are rejected by server-side admin verification."""
        _, _, regular_user = auth.register_user("regular@user.com", "Pass12345!", root=self.test_root)
        admin_user = db.get_user_by_id(DEFAULT_USER_ID, root=self.test_root)
        self.assertIsNotNone(admin_user)
        self.assertTrue(admin_user.is_admin)

        # Regular user cannot verify as admin
        self.assertFalse(auth.verify_admin(regular_user, root=self.test_root))

        # Admin user verifies as admin
        self.assertTrue(auth.verify_admin(admin_user, root=self.test_root))

        # Non-admin attempting to invoke admin APIs raises PermissionError
        with self.assertRaises(PermissionError):
            auth.admin_list_users(regular_user, root=self.test_root)

        with self.assertRaises(PermissionError):
            auth.admin_set_user_status(regular_user, target_user_id=regular_user.id, is_active=False, root=self.test_root)

        # Admin user successfully lists users and views system stats
        all_users = auth.admin_list_users(admin_user, root=self.test_root)
        self.assertGreaterEqual(len(all_users), 2)  # default admin + regular_user

        sys_stats = db.get_all_system_stats(root=self.test_root)
        self.assertIn("total_users", sys_stats)
        self.assertIn("total_notes", sys_stats)
        self.assertIn("total_captures", sys_stats)

        # Admin can deactivate a user
        act_ok = auth.admin_set_user_status(admin_user, target_user_id=regular_user.id, is_active=False, root=self.test_root)
        self.assertTrue(act_ok)

        # Deactivated user cannot log in
        login_ok, msg, sess, _ = auth.login_user("regular@user.com", "Pass12345!", root=self.test_root)
        self.assertFalse(login_ok)
        self.assertTrue("inactive" in msg.lower() or "disabled" in msg.lower())

    # -------------------------------------------------------------------------
    # Requirement 9: Preservation of Existing Records
    # -------------------------------------------------------------------------

    def test_09_existing_records_preserved_and_assigned_to_default_owner(self) -> None:
        """Requirement 9: Legacy unassigned notes/captures are preserved and assigned to admin."""
        # Query default admin user
        admin = db.get_user_by_id(DEFAULT_USER_ID, root=self.test_root)
        self.assertIsNotNone(admin)
        self.assertEqual(admin.id, DEFAULT_USER_ID)
        self.assertEqual(admin.email, "admin@secondself.ai")

        # Verify writing legacy note with default user ID persists safely
        leg_note = WikiNote(
            id="legacy_arch_01",
            raw_id="raw_leg_01",
            para="Projects",
            tags=["legacy"],
            summary="Original Note before Auth Migration",
            created="2026-11-12T00:00:00Z",
            links=[],
            body="Existing user knowledge.",
            user_id=DEFAULT_USER_ID,
        )
        storage.write_wiki_note(leg_note, root=self.test_root)

        fetched = db.get_note_by_id("legacy_arch_01", user_id=DEFAULT_USER_ID, root=self.test_root)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["user_id"], DEFAULT_USER_ID)

    # -------------------------------------------------------------------------
    # Requirement 10: RAG Q&A Isolation
    # -------------------------------------------------------------------------

    def test_10_rag_query_never_retrieves_other_users_notes(self) -> None:
        """Requirement 10: ask.retrieve_relevant_notes strictly filters candidate notes by owner."""
        _, _, user_a = auth.register_user("alice@domain.com", "Pass12345!", root=self.test_root)
        _, _, user_b = auth.register_user("bob@domain.com", "Pass12345!", root=self.test_root)

        # Alice creates a note with specific terms
        storage.write_wiki_note(
            WikiNote(
                id="alice_q_01",
                raw_id="r1",
                para="Projects",
                tags=["blockchain", "security"],
                summary="Alice Private Blockchain Research",
                created="2026-11-13T10:00:00Z",
                links=[],
                body="Zero knowledge cryptographic proofs and smart contract auditing.",
                user_id=user_a.id,
            ),
            root=self.test_root,
        )

        # Mock embeddings to return high similarity
        fake_emb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        with patch("lib.embeddings.embed_text", return_value=fake_emb), \
             patch("lib.embeddings.load_embeddings", return_value={"alice_q_01": fake_emb}):

            # When Alice searches for her notes, it is returned
            relevant_for_a = ask.retrieve_relevant_notes(
                question="blockchain cryptographic proofs",
                root=self.test_root,
                user_id=user_a.id,
            )
            self.assertTrue(any(n.id == "alice_q_01" for n, _ in relevant_for_a))

            # When Bob searches for the exact same query, Alice's note is NEVER returned
            relevant_for_b = ask.retrieve_relevant_notes(
                question="blockchain cryptographic proofs",
                root=self.test_root,
                user_id=user_b.id,
            )
            self.assertEqual(len(relevant_for_b), 0)

    # -------------------------------------------------------------------------
    # Requirement 11: Multi-Tenant Capture Isolation
    # -------------------------------------------------------------------------

    def test_11_capture_isolation_and_deduplication_scope(self) -> None:
        """Requirement 11: Captures are scoped by user_id; duplicate check is per-user."""
        _, _, user_a = auth.register_user("alice@domain.com", "Pass12345!", root=self.test_root)
        _, _, user_b = auth.register_user("bob@domain.com", "Pass12345!", root=self.test_root)

        cap_text = "Important reference document regarding system architecture."
        c_hash = storage.content_hash(cap_text)

        # Alice captures the text
        res_a = capture.capture_note(cap_text, user_id=user_a.id)
        self.assertIsInstance(res_a, CaptureResult)

        # Bob captures the identical text -> should NOT be blocked by Alice's capture
        res_b = capture.capture_note(cap_text, user_id=user_b.id)
        self.assertIsInstance(res_b, CaptureResult)
        self.assertNotEqual(res_a.id, res_b.id)

        # Alice re-checking duplicate detects her own capture
        dup_a = capture.find_duplicate_capture(c_hash, user_id=user_a.id)
        self.assertEqual(dup_a, res_a.id)

        # Bob checking duplicate detects his own capture
        dup_b = capture.find_duplicate_capture(c_hash, user_id=user_b.id)
        self.assertEqual(dup_b, res_b.id)

        # A third user with no captures sees no duplicate
        dup_c = capture.find_duplicate_capture(c_hash, user_id="unrelated-user-uuid")
        self.assertIsNone(dup_c)

        # Verify raw captures retrieval per user
        caps_a = storage.read_all_raw_captures(user_id=user_a.id, root=self.test_root)
        caps_b = storage.read_all_raw_captures(user_id=user_b.id, root=self.test_root)

        self.assertEqual(len(caps_a), 1)
        self.assertEqual(caps_a[0].id, res_a.id)
        self.assertEqual(len(caps_b), 1)
        self.assertEqual(caps_b[0].id, res_b.id)

    # -------------------------------------------------------------------------
    # Requirement 12: Password Reset Lifecycle
    # -------------------------------------------------------------------------

    def test_12_password_reset_flow(self) -> None:
        """Requirement 12: Password reset request, token validation, and password update."""
        _, _, user = auth.register_user("reset_me@domain.com", "OldPassword123!", root=self.test_root)

        # Request reset token
        ok, msg, token = auth.request_password_reset("reset_me@domain.com", root=self.test_root)
        self.assertTrue(ok)
        self.assertIsNotNone(token)

        # Apply password reset
        reset_ok, reset_msg = auth.reset_password(token, "NewSecurePassword456!", root=self.test_root)
        self.assertTrue(reset_ok, f"Reset failed: {reset_msg}")

        # Old password must fail
        old_login_ok, _, _, _ = auth.login_user("reset_me@domain.com", "OldPassword123!", root=self.test_root)
        self.assertFalse(old_login_ok)

        # New password must succeed
        new_login_ok, _, session, _ = auth.login_user("reset_me@domain.com", "NewSecurePassword456!", root=self.test_root)
        self.assertTrue(new_login_ok)
        self.assertIsNotNone(session)


if __name__ == "__main__":
    unittest.main()
