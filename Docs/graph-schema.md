# SecondSelf — Knowledge Graph Schema & Semantic Contract

**Schema Version:** `1.0.0`  
**Projection Version:** `1.0.0`  
**Entity Namespace:** `7a2e8c14-5b8d-4e9f-a63b-1d7e5f4c9a2b` (UUIDv5)  

---

## 1. Overview & Architecture

The SecondSelf Knowledge Graph transforms raw captures, autonomous PARA classifications, and dense semantic embeddings into a normalized, deterministic graph. The graph serves two primary consumers:
1. **Frontend Canvas Physics:** Force-directed network rendering in `public/index.html` via Vis.js and D3.
2. **Graph Databases:** Idempotent ingestion into Neo4j and Memgraph via Cypher statements or GraphML.

```
       [Source] (url, pdf, text)
          ▲
          │ EXTRACTED_FROM
          │
       [Item] (Note / Document) ──────BELONGS_TO─────► [PARA_Category]
          │                                            (Projects, Areas,
          ├──────TAGGED_WITH─────► [Tag] (#rag, #sop)   Resources, Archives)
          │
          └──────SEMANTIC_LINK───► [Item] (Related Note)
```

---

## 2. Canonical Identifier Strategy

To guarantee byte-identical determinism and eliminate ID fragmentation, node identifiers follow strict canonical prefixes:

| Entity Type | ID Pattern | Example | Generator Function |
| :--- | :--- | :--- | :--- |
| **Item** | `item:{record_id}` | `item:4bda912e-bfef-4e92-a7a4-28124013cba0` | `make_item_id(record_id)` |
| **PARA_Category** | `para:{category_slug}` | `para:projects`, `para:resources` | `make_para_id(category)` |
| **Tag** | `tag:{slug}` | `tag:vector-search`, `tag:sop` | `make_tag_id(tag_text)` |
| **Source** | `source:{source_type}` | `source:url`, `source:pdf`, `source:text` | `make_source_id(source_type)` |

---

## 3. Node Schema Specification

### 3.1 `Item`
Represents an individual curated note or captured knowledge atom.
- **Node Label:** `Item`
- **Required Properties:**
  - `record_id` *(string)*: Raw storage UUID / stem.
  - `title` *(string)*: Descriptive title.
  - `category` *(string)*: Assigned PARA category (`Projects`, `Areas`, `Resources`, `Archives`).
- **Optional Properties:**
  - `confidence` *(float, 0.0–1.0)*: Classifier confidence score.
  - `summary` *(string)*: 1–2 sentence takeaway.
  - `tags` *(list[string])*: Topical hashtags.
  - `source_type` *(string)*: Ingestion adapter channel (`url`, `text`, `pdf`, `markdown`).
  - `created_at` *(string, ISO-8601)*: Classification / capture timestamp.
  - `content_hash` *(string)*: SHA-256 digest of normalized text.
  - `char_count` *(int)*: Total character count.
  - `word_count` *(int)*: Total word count.
  - `needs_review` *(bool)*: Ambiguity flag if confidence $< 0.60$.
  - `file_path` *(string)*: Local path to Markdown note in `wiki/`.

### 3.2 `PARA_Category`
Fixed structural quadrant representing the core PARA organizational taxonomy.
- **Node Label:** `PARA_Category`
- **Fixed Node IDs:** `para:projects`, `para:areas`, `para:resources`, `para:archives`
- **Required Properties:**
  - `category` *(string)*: Canonical name (`Projects`, `Areas`, `Resources`, `Archives`).
  - `display_name` *(string)*: UI title.
- **Optional Properties:**
  - `target_folder` *(string)*: Destination directory in `wiki/`.
  - `description` *(string)*: Decision rubric definition.
  - `color_hex` *(string)*: Hex color code for canvas visualization:
    - Projects: `#F59E0B` (Amber)
    - Areas: `#3B82F6` (Blue)
    - Resources: `#10B981` (Green)
    - Archives: `#6B7280` (Slate)

### 3.3 `Tag`
Topical keyword or classification hashtag.
- **Node Label:** `Tag`
- **Required Properties:**
  - `name` *(string)*: Raw tag text (e.g. `vector-search`).
  - `slug` *(string)*: Normalized slug (e.g. `vector-search`).
- **Optional Properties:**
  - `frequency` *(int)*: Number of connected notes.

### 3.4 `Source`
Provenance origin or adapter channel.
- **Node Label:** `Source`
- **Required Properties:**
  - `source_type` *(string)*: Channel identifier (`url`, `pdf`, `text`, `markdown`).
  - `display_name` *(string)*: Human-readable name (e.g. `URL`, `PDF Document`).
- **Optional Properties:**
  - `source_uri` *(string)*: Web URL or local file path.

---

## 4. Edge Schema Specification

### 4.1 `BELONGS_TO`
Connects an `Item` node to its assigned `PARA_Category` quadrant.
- **Source:** `Item`
- **Target:** `PARA_Category`
- **Required Properties:**
  - `category` *(string)*: Assigned PARA category.
- **Optional Properties:**
  - `confidence` *(float)*: Classification certainty ($0.0 - 1.0$).
  - `method` *(string)*: Classification strategy (`rules`, `llm`, `hybrid`).
  - `rationale` *(string)*: Human-readable audit explanation.
  - `assigned_at` *(string, ISO-8601)*: Timestamp of classification.

### 4.2 `SEMANTIC_LINK`
Connects two `Item` nodes based on high cosine similarity between their dense 384-dimensional embeddings.
- **Source:** `Item`
- **Target:** `Item`
- **Constraint:** `source_id != target_id` (Self-loops prohibited).
- **Required Properties:**
  - `score` *(float)*: Cosine similarity score ($\ge \theta = 0.65$).
- **Optional Properties:**
  - `similarity_pct` *(int)*: Integer percentage match (e.g. `71`).
  - `method` *(string)*: Similarity algorithm (`item_to_item`).
  - `target_title` *(string)*: Title of destination note.
  - `reason` *(string)*: Explainable link justification.

### 4.3 `EXTRACTED_FROM`
Connects an `Item` node to its provenance `Source`.
- **Source:** `Item`
- **Target:** `Source`
- **Required Properties:**
  - `source_type` *(string)*: Ingestion adapter channel.
- **Optional Properties:**
  - `source_uri` *(string)*: Origin URL or filesystem path.
  - `char_count` *(int)*: Length of extracted raw content.
  - `captured_at` *(string, ISO-8601)*: Timestamp of original capture.

### 4.4 `TAGGED_WITH`
Connects an `Item` node to a `Tag` entity.
- **Source:** `Item`
- **Target:** `Tag`
- **Required Properties:**
  - `tag_slug` *(string)*: Normalized slug identifier.
- **Optional Properties:**
  - `tag_name` *(string)*: Display name.

---

## 5. Target Database Ingestion Guides

### 5.1 Neo4j & Memgraph Loading

To import the generated Cypher script into Neo4j or Memgraph:

1. **Generate Cypher Script:**
   ```powershell
   python build_graph.py --format cypher --output public/graph.cypher
   ```

2. **Execute via Cypher Shell:**
   ```bash
   cypher-shell -u neo4j -p password -f public/graph.cypher
   ```

3. **Verify Graph Ingestion:**
   ```cypher
   // Count nodes by label
   MATCH (n) RETURN labels(n)[0] AS Label, count(*) AS Count ORDER BY Count DESC;

   // Inspect PARA Clusters
   MATCH (p:PARA_Category)<-[:BELONGS_TO]-(i:Item)
   RETURN p.display_name AS Category, count(i) AS Notes;

   // Trace Semantic Knowledge Links
   MATCH (a:Item)-[r:SEMANTIC_LINK]->(b:Item)
   RETURN a.title, r.score, b.title ORDER BY r.score DESC;
   ```

### 5.2 Frontend Canvas Physics (Vis.js / D3)

The graph generator emits `public/graph.json` in node-link JSON format directly consumable by Vis.js:

```javascript
fetch('/graph.json')
  .then(res => res.json())
  .then(graphData => {
    const container = document.getElementById('network-canvas');
    const data = {
      nodes: new vis.DataSet(graphData.nodes),
      edges: new vis.DataSet(graphData.edges)
    };
    const options = {
      nodes: {
        shape: 'dot',
        scaling: { min: 12, max: 36 },
        font: { color: '#E2E8F0', size: 14, face: 'Inter' }
      },
      groups: {
        Projects: { color: { background: '#F59E0B', border: '#D97706' } },
        Areas: { color: { background: '#3B82F6', border: '#2563EB' } },
        Resources: { color: { background: '#10B981', border: '#059669' } },
        Archives: { color: { background: '#6B7280', border: '#4B5563' } }
      },
      physics: {
        solver: 'barnesHut',
        barnesHut: { gravitationalConstant: -3000, springLength: 95 }
      }
    };
    new vis.Network(container, data, options);
  });
```
