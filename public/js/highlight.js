/**
 * SecondSelf — Graph Citation Highlighting & Focus Engine (Task 4.3)
 *
 * Visually bridges conversational answer citations with the knowledge graph:
 * - Maps citations to graph nodes with flexible prefix and property matching
 * - Highlights cited nodes with high-contrast outlines and elevated scale
 * - Dims non-cited nodes and edges to isolate visual context
 * - Auto-fits the camera to the cited subgraph
 * - Supports reciprocal focus: clicking a citation opens its DetailPanel and centers the graph
 * - Satisfies WCAG 2.1 AA by altering geometry (scale), borders, and contrast (not color alone)
 */

/**
 * Resolves a citation object to its corresponding node in the graph.
 *
 * @param {Array<Object>} allNodes Array of node definitions from graph.json
 * @param {Object} citation Citation object from /api/ask response
 * @returns {Object|null} Matching node or null
 */
export function findNodeForCitation(allNodes, citation) {
  if (!allNodes || !citation) return null;

  const targetId = citation.item_id || '';
  const strippedId = targetId.replace(/^item:/, '');
  const title = (citation.title || '').trim().toLowerCase();

  return (
    allNodes.find((n) => {
      if (n.id === targetId || n.id === strippedId) return true;
      if (n.id.replace(/^item:/, '') === strippedId) return true;
      if (n.properties?.record_id === targetId || n.properties?.record_id === strippedId) return true;
      if (title && (n.label || '').trim().toLowerCase() === title) return true;
      return false;
    }) || null
  );
}

/**
 * Highlights cited nodes in the graph and dims all non-cited nodes and edges.
 *
 * @param {Object} graphRenderer The GraphRenderer instance (Vis.js / Canvas)
 * @param {Array<Object>} citations Citations array from /api/ask response
 * @param {Array<Object>} allNodes Full nodes array
 * @param {Array<Object>} allEdges Full edges array
 * @returns {Set<string>} Set of matched cited node IDs
 */
export function highlightCitations(graphRenderer, citations, allNodes, allEdges) {
  if (!graphRenderer || !citations || citations.length === 0) {
    return new Set();
  }

  // 1. Identify all cited node IDs
  const citedNodeIds = new Set();
  const citationMap = new Map(); // nodeId -> citation

  citations.forEach((c) => {
    const node = findNodeForCitation(allNodes, c);
    if (node) {
      citedNodeIds.add(node.id);
      citationMap.set(node.id, c);
    }
  });

  if (citedNodeIds.size === 0) {
    console.warn('[Highlight] None of the cited sources were found in active graph nodes');
    return new Set();
  }

  // 2. Identify connecting edges between cited nodes
  const connectingEdgeIds = new Set();
  const visibleEdges = graphRenderer.visibleEdges || allEdges || [];
  const visibleNodes = graphRenderer.visibleNodes || allNodes || [];

  visibleEdges.forEach((edge) => {
    const fromCited = citedNodeIds.has(edge.from);
    const toCited = citedNodeIds.has(edge.to);
    const edgeId = `${edge.from}->${edge.to}:${edge.type}`;

    if (fromCited || toCited) {
      connectingEdgeIds.add(edgeId);
    }
  });

  // 3. Update Vis.js Dataset if network is active
  if (graphRenderer.visData && graphRenderer.visData.nodes) {
    const updatedNodes = visibleNodes.map((n) => {
      const isCited = citedNodeIds.has(n.id);
      const group = n.properties?.category || n.group || 'Default';
      const colorToken = graphRenderer.colors[group] || graphRenderer.colors[n.type] || graphRenderer.colors.Default;

      if (isCited) {
        // Multi-channel highlight: doubled size + white border + cyan glow + high contrast font
        return {
          id: n.id,
          size: 26,
          borderWidth: 4,
          color: {
            background: colorToken.bg,
            border: '#ffffff',
            highlight: {
              background: colorToken.highlight,
              border: '#ffffff'
            }
          },
          font: {
            size: 15,
            color: '#ffffff',
            bold: true,
            strokeWidth: 4,
            strokeColor: '#090d16'
          },
          shadow: {
            enabled: true,
            color: '#38bdf8',
            size: 18,
            x: 0,
            y: 0
          }
        };
      } else {
        // Dimmed state: reduced opacity and zero shadow
        return {
          id: n.id,
          size: n.value || 14,
          borderWidth: 1,
          color: {
            background: 'rgba(30, 41, 59, 0.22)',
            border: 'rgba(51, 65, 85, 0.18)'
          },
          font: {
            size: 12,
            color: 'rgba(148, 163, 184, 0.18)'
          },
          shadow: { enabled: false }
        };
      }
    });

    const updatedEdges = visibleEdges.map((e) => {
      const edgeId = `${e.from}->${e.to}:${e.type}`;
      const isConnected = connectingEdgeIds.has(edgeId);
      const colorToken = graphRenderer.edgeColors[e.type] || graphRenderer.edgeColors.DEFAULT;

      return {
        id: edgeId,
        color: {
          color: isConnected ? 'rgba(56, 189, 248, 0.85)' : 'rgba(30, 41, 59, 0.05)'
        },
        width: isConnected ? 2.5 : 0.8
      };
    });

    graphRenderer.visData.nodes.update(updatedNodes);
    graphRenderer.visData.edges.update(updatedEdges);

    // 4. Auto-fit camera to the cited subgraph
    try {
      graphRenderer.network.fit({
        nodes: Array.from(citedNodeIds),
        animation: {
          duration: 600,
          easingFunction: 'easeInOutQuad'
        }
      });
    } catch (fitErr) {
      console.warn('[Highlight] Could not auto-fit camera:', fitErr);
    }
  }

  return citedNodeIds;
}

/**
 * Restores graph visualization to default vibrancy and fits view.
 *
 * @param {Object} graphRenderer
 */
export function clearCitations(graphRenderer) {
  if (!graphRenderer) return;

  if (typeof graphRenderer.resetHighlight === 'function') {
    graphRenderer.resetHighlight();
  }

  if (graphRenderer.network && typeof graphRenderer.fit === 'function') {
    graphRenderer.fit();
  }
}

/**
 * Focuses camera on a citation's node and opens its slide-out detail panel.
 *
 * @param {Object} graphRenderer
 * @param {string} nodeId Target node ID
 * @param {Object} detailPanel DetailPanel instance
 * @param {Array<Object>} allNodes Full nodes array
 */
export function focusCitation(graphRenderer, nodeId, detailPanel, allNodes) {
  if (!nodeId || !graphRenderer) return;

  // Center & focus graph
  graphRenderer.focusNode(nodeId);

  // Open note details if panel is available
  if (detailPanel && allNodes) {
    const node = allNodes.find((n) => n.id === nodeId || n.id.replace(/^item:/, '') === nodeId.replace(/^item:/, ''));
    if (node) {
      detailPanel.open(node);
    }
  }
}
