/**
 * SecondSelf — Data Loader & Schema Validator (Task 3.2)
 *
 * Fetches graph.json artifacts with multi-path fallback, defensively validates
 * node-link schema, and normalizes data structures for graph & list views.
 */

export class GraphDataLoader {
  /**
   * Primary candidate URLs in priority order.
   */
  static CANDIDATE_URLS = [
    'data/graph.json',
    'graph.json',
    './data/graph.json',
    './graph.json'
  ];

  /**
   * Loads graph data from the first working candidate URL.
   *
   * @param {string|null} customUrl Optional explicit URL to load from.
   * @returns {Promise<{metadata: Object, statistics: Object, nodes: Array, edges: Array, raw: Object}>}
   */
  static async load(customUrl = null) {
    const urls = customUrl ? [customUrl, ...GraphDataLoader.CANDIDATE_URLS] : GraphDataLoader.CANDIDATE_URLS;
    let lastError = null;

    for (const url of urls) {
      try {
        const response = await fetch(url, {
          headers: { 'Accept': 'application/json' },
          cache: 'no-cache'
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status} (${response.statusText})`);
        }

        const rawData = await response.json();
        const validated = GraphDataLoader.validateAndNormalize(rawData, url);
        return validated;
      } catch (err) {
        lastError = err;
        // Continue trying remaining candidate URLs
      }
    }

    throw new Error(
      `Failed to load SecondSelf graph artifact from candidate paths (${urls.join(', ')}). Reason: ${lastError ? lastError.message : 'Unknown error'}`
    );
  }

  /**
   * Defensively validates the loaded JSON object against the Task 3.1 schema.
   * Normalizes missing optional fields without throwing unnecessary crashes.
   *
   * @param {any} data Parsed JSON payload
   * @param {string} sourceUrl URL from which data was retrieved
   * @returns {{metadata: Object, statistics: Object, nodes: Array, edges: Array, raw: Object}}
   */
  static validateAndNormalize(data, sourceUrl) {
    if (!data || typeof data !== 'object') {
      throw new Error(`Invalid graph payload: root must be a JSON object, got ${typeof data}`);
    }

    // 1. Validate & extract nodes
    if (!Array.isArray(data.nodes)) {
      throw new Error("Invalid graph payload: 'nodes' must be an array");
    }

    if (data.nodes.length === 0) {
      throw new Error("Empty graph: 'nodes' array contains 0 elements");
    }

    const normalizedNodes = [];
    const nodeIds = new Set();

    for (let i = 0; i < data.nodes.length; i++) {
      const n = data.nodes[i];
      if (!n || typeof n !== 'object') {
        console.warn(`[DataLoader] Skipping malformed node at index ${i}:`, n);
        continue;
      }

      if (!n.id) {
        console.warn(`[DataLoader] Skipping node at index ${i} missing required 'id':`, n);
        continue;
      }

      const id = String(n.id);
      if (nodeIds.has(id)) {
        console.warn(`[DataLoader] Duplicate node id '${id}' detected at index ${i}, preserving first occurrence`);
        continue;
      }
      nodeIds.add(id);

      // Normalization of node attributes
      const type = n.type || 'Item';
      const label = n.label || n.title || id;
      const group = n.group || n.category || (type === 'Item' ? 'Resources' : type);
      const degree = typeof n.degree === 'number' ? n.degree : 0;
      const value = typeof n.value === 'number' ? n.value : Math.max(12, Math.min(36, 12 + degree * 2));
      const summary = n.summary || (n.properties && n.properties.summary) || '';
      const tags = Array.isArray(n.tags) ? n.tags : (n.properties && Array.isArray(n.properties.tags) ? n.properties.tags : []);
      const properties = (n.properties && typeof n.properties === 'object') ? n.properties : {};

      normalizedNodes.push({
        id,
        label,
        type,
        group,
        degree,
        value,
        summary,
        tags,
        properties: {
          ...properties,
          title: properties.title || label,
          category: properties.category || group,
          confidence: typeof properties.confidence === 'number' ? properties.confidence : null,
          rationale: properties.rationale || '',
          source_type: properties.source_type || 'markdown',
          created_at: properties.created_at || null,
          record_id: properties.record_id || id.replace(/^[a-z_]+:/i, ''),
          needs_review: Boolean(properties.needs_review)
        }
      });
    }

    // 2. Validate & extract edges (supports dual 'edges' and 'links' keys)
    const rawEdges = Array.isArray(data.edges) ? data.edges : (Array.isArray(data.links) ? data.links : []);
    const normalizedEdges = [];
    const edgeKeys = new Set();

    for (let i = 0; i < rawEdges.length; i++) {
      const e = rawEdges[i];
      if (!e || typeof e !== 'object') continue;

      const source = String(e.source || e.from || '');
      const target = String(e.target || e.to || '');

      if (!source || !target) {
        console.warn(`[DataLoader] Skipping edge at index ${i} with missing endpoints:`, e);
        continue;
      }

      // Check endpoints exist in nodes
      if (!nodeIds.has(source) || !nodeIds.has(target)) {
        // Tolerantly skip dangling edges without crashing the frontend
        console.warn(`[DataLoader] Dropping edge (${source} -> ${target}): endpoint node not in graph`);
        continue;
      }

      const type = e.type || 'RELATED_TO';
      const edgeKey = `${source}->${target}:${type}`;
      if (edgeKeys.has(edgeKey)) continue;
      edgeKeys.add(edgeKey);

      normalizedEdges.push({
        from: source,
        to: target,
        source,
        target,
        type,
        value: typeof e.value === 'number' ? e.value : 1.0,
        title: e.title || type,
        properties: (e.properties && typeof e.properties === 'object') ? e.properties : {}
      });
    }

    // 3. Metadata & Statistics extraction
    const metadata = {
      schema_version: data.schema_version || '1.0.0',
      projection_version: data.projection_version || '1.0.0',
      generated_at: data.generated_at || new Date().toISOString(),
      corpus_hash: data.corpus_hash || '',
      source_url: sourceUrl
    };

    const stats = data.statistics || {};
    const statistics = {
      total_nodes: normalizedNodes.length,
      total_edges: normalizedEdges.length,
      node_counts_by_type: stats.node_counts_by_type || GraphDataLoader.computeTypeCounts(normalizedNodes),
      edge_counts_by_type: stats.edge_counts_by_type || {},
      avg_degree: typeof stats.avg_degree === 'number' ? stats.avg_degree : (normalizedEdges.length * 2 / (normalizedNodes.length || 1)),
      graph_density: stats.graph_density || 0
    };

    return {
      metadata,
      statistics,
      nodes: normalizedNodes,
      edges: normalizedEdges,
      raw: data
    };
  }

  /**
   * Computes fallback node counts by type.
   */
  static computeTypeCounts(nodes) {
    const counts = {};
    for (const n of nodes) {
      counts[n.type] = (counts[n.type] || 0) + 1;
    }
    return counts;
  }
}
