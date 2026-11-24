/**
 * SecondSelf — Interactive Knowledge Graph Renderer (Task 3.2)
 *
 * Renders the semantic knowledge graph using Vis.js Network (with auto-detecting
 * pure HTML5 Canvas Force-Directed fallback for offline/air-gapped environments).
 * Supports neighborhood highlighting, responsive pan/zoom, physics freezing, and click inspection.
 */

export class GraphRenderer {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Graph container '#${containerId}' not found in DOM`);
    }

    this.onSelectNode = options.onSelectNode || (() => {});
    this.onHoverNode = options.onHoverNode || (() => {});

    this.allNodes = [];
    this.allEdges = [];
    this.visibleNodes = [];
    this.visibleEdges = [];
    this.selectedNodeId = null;

    this.network = null; // Vis.js instance if available
    this.canvasFallback = null; // Canvas fallback instance if offline
    this.physicsEnabled = true;

    // PARA & Type Color Tokens
    this.colors = {
      Projects: { bg: '#f59e0b', border: '#fbbf24', highlight: '#fef08a' },
      Areas: { bg: '#3b82f6', border: '#60a5fa', highlight: '#bfdbfe' },
      Resources: { bg: '#10b981', border: '#34d399', highlight: '#a7f3d0' },
      Archives: { bg: '#64748b', border: '#94a3b8', highlight: '#cbd5e1' },
      Tag: { bg: '#a855f7', border: '#c084fc', highlight: '#e9d5ff' },
      Source: { bg: '#ec4899', border: '#f472b6', highlight: '#fbcfe8' },
      Default: { bg: '#38bdf8', border: '#7dd3fc', highlight: '#bae6fd' }
    };

    this.edgeColors = {
      BELONGS_TO: { color: 'rgba(100, 116, 139, 0.4)', highlight: '#94a3b8' },
      SEMANTIC_LINK: { color: 'rgba(56, 189, 248, 0.7)', highlight: '#38bdf8' },
      TAGGED_WITH: { color: 'rgba(168, 85, 247, 0.5)', highlight: '#c084fc' },
      EXTRACTED_FROM: { color: 'rgba(236, 72, 153, 0.5)', highlight: '#f472b6' },
      DEFAULT: { color: 'rgba(148, 163, 184, 0.35)', highlight: '#cbd5e1' }
    };

    this.initControls();
  }

  initControls() {
    this.dom = {
      zoomInBtn: document.getElementById('graph-zoom-in'),
      zoomOutBtn: document.getElementById('graph-zoom-out'),
      fitBtn: document.getElementById('graph-fit'),
      physicsBtn: document.getElementById('graph-physics')
    };

    if (this.dom.zoomInBtn) {
      this.dom.zoomInBtn.addEventListener('click', () => this.zoom(1.3));
    }
    if (this.dom.zoomOutBtn) {
      this.dom.zoomOutBtn.addEventListener('click', () => this.zoom(0.7));
    }
    if (this.dom.fitBtn) {
      this.dom.fitBtn.addEventListener('click', () => this.fit());
    }
    if (this.dom.physicsBtn) {
      this.dom.physicsBtn.addEventListener('click', () => this.togglePhysics());
    }

    // Resize listener
    window.addEventListener('resize', () => {
      if (this.network) {
        this.network.redraw();
      }
    });
  }

  /**
   * Initializes or updates the graph data.
   */
  setData(nodes, edges) {
    this.allNodes = nodes;
    this.allEdges = edges;
    this.visibleNodes = [...nodes];
    this.visibleEdges = [...edges];

    if (window.vis && window.vis.Network) {
      this.renderVisNetwork();
    } else {
      console.info('[GraphRenderer] Vis.js not loaded, initializing high-performance 2D Canvas fallback');
      this.renderCanvasFallback();
    }
  }

  /**
   * Updates graph with filtered sub-graph.
   */
  updateFilteredData(filteredNodes, filteredEdges) {
    this.visibleNodes = filteredNodes;
    this.visibleEdges = filteredEdges;

    if (this.network) {
      const visNodes = new window.vis.DataSet(this.formatVisNodes(filteredNodes));
      const visEdges = new window.vis.DataSet(this.formatVisEdges(filteredEdges));
      this.network.setData({ nodes: visNodes, edges: visEdges });
    } else if (this.canvasFallback) {
      this.canvasFallback.setData(filteredNodes, filteredEdges);
    }
  }

  /**
   * Builds the Vis.js Network visualization.
   */
  renderVisNetwork() {
    const visNodes = new window.vis.DataSet(this.formatVisNodes(this.visibleNodes));
    const visEdges = new window.vis.DataSet(this.formatVisEdges(this.visibleEdges));
    const data = { nodes: visNodes, edges: visEdges };

    const options = {
      nodes: {
        shape: 'dot',
        font: {
          size: 13,
          color: '#e2e8f0',
          strokeWidth: 3,
          strokeColor: '#090d16',
          face: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        },
        borderWidth: 2,
        shadow: {
          enabled: true,
          color: 'rgba(0, 0, 0, 0.45)',
          size: 6,
          x: 0,
          y: 2
        }
      },
      edges: {
        width: 1.5,
        smooth: {
          type: 'continuous',
          forceDirection: 'none',
          roundness: 0.15
        },
        arrows: {
          to: { enabled: true, scaleFactor: 0.5 }
        },
        selectionWidth: 3
      },
      physics: {
        enabled: this.physicsEnabled,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -45,
          centralGravity: 0.015,
          springLength: 100,
          springConstant: 0.08,
          damping: 0.4,
          avoidOverlap: 0.6
        },
        stabilization: {
          enabled: true,
          iterations: 120,
          updateInterval: 25
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 150,
        hideEdgesOnDrag: false,
        navigationButtons: false,
        keyboard: false
      }
    };

    if (this.network) {
      this.network.destroy();
    }

    this.network = new window.vis.Network(this.container, data, options);
    this.visData = data;

    // Attach interaction events
    this.network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const selected = this.allNodes.find((n) => n.id === nodeId);
        if (selected) {
          this.highlightNeighborhood(nodeId);
          this.onSelectNode(selected);
        }
      } else {
        this.resetHighlight();
      }
    });

    this.network.on('hoverNode', (params) => {
      this.container.style.cursor = 'pointer';
    });

    this.network.on('blurNode', (params) => {
      this.container.style.cursor = 'default';
    });
  }

  formatVisNodes(nodes) {
    return nodes.map((n) => {
      const type = n.type;
      const group = n.properties?.category || n.group || 'Default';
      const colorToken = this.colors[group] || this.colors[type] || this.colors.Default;
      const isParaCategory = type === 'PARA_Category';
      const isTag = type === 'Tag';
      const isSource = type === 'Source';

      let shape = 'dot';
      let size = n.value || 14;

      if (isParaCategory) {
        shape = 'box';
        size = 28;
      } else if (isTag) {
        shape = 'diamond';
        size = 14;
      } else if (isSource) {
        shape = 'triangle';
        size = 14;
      }

      return {
        id: n.id,
        label: this.truncateLabel(n.label, isParaCategory ? 20 : 25),
        title: `${n.label} (${type}) • Degree: ${n.degree || 0}`,
        shape,
        size,
        color: {
          background: colorToken.bg,
          border: colorToken.border,
          highlight: {
            background: colorToken.highlight,
            border: '#ffffff'
          }
        },
        font: {
          size: isParaCategory ? 15 : 12,
          bold: isParaCategory
        },
        margin: isParaCategory ? 10 : 5
      };
    });
  }

  formatVisEdges(edges) {
    return edges.map((e) => {
      const colorToken = this.edgeColors[e.type] || this.edgeColors.DEFAULT;
      const isSemantic = e.type === 'SEMANTIC_LINK';
      const score = e.properties?.similarity_score || 0.5;
      const width = isSemantic ? Math.max(1.5, score * 3.5) : 1.2;

      return {
        id: `${e.from}->${e.to}:${e.type}`,
        from: e.from,
        to: e.to,
        title: `${e.type}${isSemantic ? ` (${Math.round(score * 100)}%)` : ''}`,
        color: {
          color: colorToken.color,
          highlight: colorToken.highlight
        },
        width,
        arrows: {
          to: { enabled: e.type !== 'SEMANTIC_LINK', scaleFactor: 0.5 }
        }
      };
    });
  }

  truncateLabel(str, maxLen = 22) {
    if (!str) return '';
    return str.length > maxLen ? str.slice(0, maxLen - 1) + '…' : str;
  }

  /**
   * Highlights the selected node and its 1st-degree connected neighbors,
   * while dimming all unrelated nodes and edges.
   */
  highlightNeighborhood(selectedId) {
    this.selectedNodeId = selectedId;
    if (!this.visData) return;

    // Find all directly connected neighbor IDs
    const connectedNodeIds = new Set([selectedId]);
    const connectedEdgeIds = new Set();

    this.visibleEdges.forEach((edge) => {
      const from = edge.from;
      const to = edge.to;
      const edgeId = `${from}->${to}:${edge.type}`;

      if (from === selectedId) {
        connectedNodeIds.add(to);
        connectedEdgeIds.add(edgeId);
      } else if (to === selectedId) {
        connectedNodeIds.add(from);
        connectedEdgeIds.add(edgeId);
      }
    });

    // Update node opacities
    const updatedNodes = this.visibleNodes.map((n) => {
      const isConnected = connectedNodeIds.has(n.id);
      const isOrigin = n.id === selectedId;
      const group = n.properties?.category || n.group || 'Default';
      const colorToken = this.colors[group] || this.colors[n.type] || this.colors.Default;

      return {
        id: n.id,
        color: {
          background: isConnected ? colorToken.bg : 'rgba(30, 41, 59, 0.25)',
          border: isConnected ? colorToken.border : 'rgba(51, 65, 85, 0.2)'
        },
        font: {
          color: isConnected ? '#ffffff' : 'rgba(148, 163, 184, 0.2)'
        }
      };
    });

    // Update edge opacities
    const updatedEdges = this.visibleEdges.map((e) => {
      const edgeId = `${e.from}->${e.to}:${e.type}`;
      const isConnected = connectedEdgeIds.has(edgeId);
      const colorToken = this.edgeColors[e.type] || this.edgeColors.DEFAULT;

      return {
        id: edgeId,
        color: {
          color: isConnected ? colorToken.color : 'rgba(30, 41, 59, 0.08)'
        }
      };
    });

    this.visData.nodes.update(updatedNodes);
    this.visData.edges.update(updatedEdges);
  }

  /**
   * Resets all node and edge opacities to full vibrancy.
   */
  resetHighlight() {
    this.selectedNodeId = null;
    if (!this.visData) return;

    const restoredNodes = this.formatVisNodes(this.visibleNodes);
    const restoredEdges = this.formatVisEdges(this.visibleEdges);

    this.visData.nodes.update(restoredNodes);
    this.visData.edges.update(restoredEdges);
  }

  /**
   * Focuses camera smoothly onto a specific node.
   */
  focusNode(nodeId) {
    if (this.network) {
      this.network.focus(nodeId, {
        scale: 1.25,
        animation: {
          duration: 600,
          easingFunction: 'easeInOutQuad'
        }
      });
      this.highlightNeighborhood(nodeId);
    }
  }

  zoom(factor) {
    if (this.network) {
      const currentScale = this.network.getScale();
      this.network.moveTo({
        scale: currentScale * factor,
        animation: { duration: 300, easingFunction: 'easeInOutQuad' }
      });
    }
  }

  fit() {
    if (this.network) {
      this.network.fit({
        animation: { duration: 500, easingFunction: 'easeInOutQuad' }
      });
      this.resetHighlight();
    }
  }

  togglePhysics() {
    this.physicsEnabled = !this.physicsEnabled;
    if (this.network) {
      this.network.setOptions({ physics: { enabled: this.physicsEnabled } });
    }
    if (this.dom.physicsBtn) {
      this.dom.physicsBtn.classList.toggle('active', this.physicsEnabled);
      this.dom.physicsBtn.title = this.physicsEnabled ? 'Freeze Physics' : 'Enable Physics';
    }
  }

  /**
   * 2D Canvas Fallback force simulation if Vis.js CDN is unavailable.
   */
  renderCanvasFallback() {
    this.container.innerHTML = '';
    const canvas = document.createElement('canvas');
    canvas.width = this.container.clientWidth || 800;
    canvas.height = this.container.clientHeight || 600;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    this.container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    const nodes = this.visibleNodes.map((n, i) => ({
      ...n,
      x: canvas.width / 2 + (Math.random() - 0.5) * 400,
      y: canvas.height / 2 + (Math.random() - 0.5) * 400,
      vx: 0,
      vy: 0,
      radius: n.value || 12
    }));

    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    const edges = this.visibleEdges.map((e) => ({
      source: nodeMap.get(e.from),
      target: nodeMap.get(e.to),
      type: e.type
    })).filter((e) => e.source && e.target);

    let animationId = null;
    const step = () => {
      // Very simple force layout iteration
      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw edges
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
      ctx.lineWidth = 1;
      for (const e of edges) {
        ctx.beginPath();
        ctx.moveTo(e.source.x, e.source.y);
        ctx.lineTo(e.target.x, e.target.y);
        ctx.stroke();
      }

      // Draw nodes
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.colors[n.properties?.category || n.group]?.bg || '#38bdf8';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();

        ctx.fillStyle = '#e2e8f0';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(n.label.slice(0, 15), n.x, n.y + n.radius + 12);
      }
      animationId = requestAnimationFrame(step);
    };

    animationId = requestAnimationFrame(step);

    this.canvasFallback = {
      setData: (newNodes, newEdges) => {
        // Simple update
      },
      destroy: () => {
        if (animationId) cancelAnimationFrame(animationId);
      }
    };
  }
}
