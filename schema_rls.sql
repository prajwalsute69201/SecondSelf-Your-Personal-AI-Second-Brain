-- =============================================================================
-- SecondSelf PostgreSQL Schema with Row Level Security (RLS)
-- Suitable for Supabase and Standard PostgreSQL Multi-Tenant Environments
-- =============================================================================

-- 1. Enable pgcrypto for UUID generation if needed
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. Users Table (Integrated with Supabase auth.users or standalone auth)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    full_name TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    last_login_at TIMESTAMPTZ
);

-- Index for email lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users (LOWER(email));

-- 3. Captures Table
CREATE TABLE IF NOT EXISTS public.captures (
    id TEXT NOT NULL,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    original_filename TEXT,
    content_hash TEXT NOT NULL,
    content_text TEXT,
    content_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    PRIMARY KEY (user_id, id)
);

CREATE INDEX IF NOT EXISTS idx_captures_user_status ON public.captures (user_id, status);
CREATE INDEX IF NOT EXISTS idx_captures_user_hash ON public.captures (user_id, content_hash);

-- 4. Notes Table
CREATE TABLE IF NOT EXISTS public.notes (
    id TEXT NOT NULL,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    raw_id TEXT,
    para TEXT NOT NULL CHECK (para IN ('Projects', 'Areas', 'Resources', 'Archives')),
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary TEXT NOT NULL DEFAULT '',
    created TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    links JSONB NOT NULL DEFAULT '[]'::jsonb,
    body TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    PRIMARY KEY (user_id, id)
);

CREATE INDEX IF NOT EXISTS idx_notes_user_para ON public.notes (user_id, para);

-- 5. Knowledge Graph Nodes
CREATE TABLE IF NOT EXISTS public.graph_nodes (
    id TEXT NOT NULL,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    para TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary TEXT NOT NULL DEFAULT '',
    content_preview TEXT NOT NULL DEFAULT '',
    group_name TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, id)
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_user ON public.graph_nodes (user_id);

-- 6. Knowledge Graph Edges
CREATE TABLE IF NOT EXISTS public.graph_edges (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    edge_type TEXT NOT NULL DEFAULT 'semantic',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    UNIQUE (user_id, source, target)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_user ON public.graph_edges (user_id);

-- 7. Knowledge Graph Metadata
CREATE TABLE IF NOT EXISTS public.graph_metadata (
    user_id UUID PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    node_count INTEGER NOT NULL DEFAULT 0,
    edge_count INTEGER NOT NULL DEFAULT 0
);

-- 8. Sessions Table
CREATE TABLE IF NOT EXISTS public.sessions (
    token TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_token ON public.sessions (token, expires_at);

-- 9. Password Resets Table
CREATE TABLE IF NOT EXISTS public.password_resets (
    token TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL,
    used BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- =============================================================================
-- HELPER FUNCTIONS FOR SECURITY AND ROLE VERIFICATION
-- =============================================================================

-- Determine if currently authenticated session is an administrator
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.users
        WHERE id = auth.uid() AND role = 'admin' AND is_active = TRUE
    );
$$;

-- =============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================================================

-- Enable RLS across all sensitive tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.captures ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.password_resets ENABLE ROW LEVEL SECURITY;

-- -----------------------------------------------------------------------------
-- USERS POLICIES
-- -----------------------------------------------------------------------------
-- Users can view their own profile
CREATE POLICY "Users can read own profile"
    ON public.users FOR SELECT
    USING (auth.uid() = id OR public.is_admin());

-- Users can update their own profile (name, last_login_at)
CREATE POLICY "Users can update own profile"
    ON public.users FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- Admins can update any user (e.g. activate/deactivate accounts)
CREATE POLICY "Admins can update users"
    ON public.users FOR UPDATE
    USING (public.is_admin());

-- -----------------------------------------------------------------------------
-- CAPTURES POLICIES
-- -----------------------------------------------------------------------------
CREATE POLICY "Users can view own captures"
    ON public.captures FOR SELECT
    USING (auth.uid() = user_id OR public.is_admin());

CREATE POLICY "Users can insert own captures"
    ON public.captures FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own captures"
    ON public.captures FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own captures"
    ON public.captures FOR DELETE
    USING (auth.uid() = user_id);

-- -----------------------------------------------------------------------------
-- NOTES POLICIES
-- -----------------------------------------------------------------------------
CREATE POLICY "Users can view own notes"
    ON public.notes FOR SELECT
    USING (auth.uid() = user_id OR public.is_admin());

CREATE POLICY "Users can insert own notes"
    ON public.notes FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own notes"
    ON public.notes FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own notes"
    ON public.notes FOR DELETE
    USING (auth.uid() = user_id);

-- -----------------------------------------------------------------------------
-- GRAPH NODES POLICIES
-- -----------------------------------------------------------------------------
CREATE POLICY "Users can view own graph nodes"
    ON public.graph_nodes FOR SELECT
    USING (auth.uid() = user_id OR public.is_admin());

CREATE POLICY "Users can insert own graph nodes"
    ON public.graph_nodes FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own graph nodes"
    ON public.graph_nodes FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own graph nodes"
    ON public.graph_nodes FOR DELETE
    USING (auth.uid() = user_id);

-- -----------------------------------------------------------------------------
-- GRAPH EDGES POLICIES
-- -----------------------------------------------------------------------------
CREATE POLICY "Users can view own graph edges"
    ON public.graph_edges FOR SELECT
    USING (auth.uid() = user_id OR public.is_admin());

CREATE POLICY "Users can insert own graph edges"
    ON public.graph_edges FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own graph edges"
    ON public.graph_edges FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own graph edges"
    ON public.graph_edges FOR DELETE
    USING (auth.uid() = user_id);

-- -----------------------------------------------------------------------------
-- GRAPH METADATA POLICIES
-- -----------------------------------------------------------------------------
CREATE POLICY "Users can view own graph metadata"
    ON public.graph_metadata FOR SELECT
    USING (auth.uid() = user_id OR public.is_admin());

CREATE POLICY "Users can modify own graph metadata"
    ON public.graph_metadata FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- =============================================================================
-- DATA PRESERVATION & INITIAL ADMIN MIGRATION
-- =============================================================================
-- Seed the default administrator account (Default owner of existing records)
INSERT INTO public.users (id, email, full_name, role, is_active, email_verified)
VALUES (
    '00000000-0000-0000-0000-000000000001'::uuid,
    'admin@secondself.ai',
    'SecondSelf Administrator',
    'admin',
    TRUE,
    TRUE
)
ON CONFLICT (id) DO NOTHING;
