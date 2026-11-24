/**
 * SecondSelf — Application Entrypoint (Tasks 3.2 & 4.3)
 *
 * Bootstraps the static frontend, loads graph artifacts, coordinates
 * the reactive filter engine, canvas graph renderer, slide-out detail panel,
 * and the conversational Q&A search engine with graph citation highlighting.
 */

import { GraphDataLoader } from './dataLoader.js';
import { GraphRenderer } from './graph.js';
import { FilterEngine } from './filters.js';
import { DetailPanel } from './detailPanel.js';
import { SearchController } from './search.js';
import { renderAnswer, clearAnswer, setActiveCitationCard } from './citations.js';
import { highlightCitations, clearCitations, focusCitation, findNodeForCitation } from './highlight.js';

class SecondSelfApp {
  constructor() {
    this.graphData = null;
    this.graphRenderer = null;
    this.filterEngine = null;
    this.detailPanel = null;
    this.searchController = null;
    this.currentAnswerData = null;

    this.initDOM();
  }

  initDOM() {
    this.dom = {
      loadingOverlay: document.getElementById('loading-overlay'),
      loadingMsg: document.getElementById('loading-msg'),
      errorOverlay: document.getElementById('error-overlay'),
      errorMsg: document.getElementById('error-msg'),
      retryBtn: document.getElementById('retry-btn'),
      statsStrip: document.getElementById('stats-strip'),
      statTotal: document.getElementById('stat-total'),
      statProjects: document.getElementById('stat-projects'),
      statAreas: document.getElementById('stat-areas'),
      statResources: document.getElementById('stat-resources'),
      statArchives: document.getElementById('stat-archives'),
      statEdges: document.getElementById('stat-edges'),
      answerContainer: document.getElementById('qa-answer-container')
    };

    if (this.dom.retryBtn) {
      this.dom.retryBtn.addEventListener('click', () => this.start());
    }
  }

  async start() {
    this.showLoading('Loading SecondSelf knowledge graph...');

    try {
      // 1. Fetch & Validate Data
      this.graphData = await GraphDataLoader.load();
      console.info(
        `[App] Successfully loaded graph with ${this.graphData.nodes.length} nodes and ${this.graphData.edges.length} edges (schema v${this.graphData.metadata.schema_version})`
      );

      // 2. Populate Header Stats Strip
      this.updateHeaderStats(this.graphData);

      // 3. Initialize Detail Panel
      this.detailPanel = new DetailPanel(this.graphData.nodes, this.graphData.edges, {
        onSelectNode: (node) => {
          this.graphRenderer?.focusNode(node.id);
          this.detailPanel.open(node);
          // Reciprocal sync: highlight matching citation card if active
          if (this.currentAnswerData?.citations) {
            setActiveCitationCard(this.dom.answerContainer, node.id);
          }
        },
        onFilterByTag: (tag) => {
          if (this.filterEngine) {
            const input = document.getElementById('search-input');
            if (input) {
              input.value = tag;
              input.dispatchEvent(new Event('input'));
            }
          }
        }
      });

      // 4. Initialize Graph Renderer
      this.graphRenderer = new GraphRenderer('network-canvas', {
        onSelectNode: (node) => {
          this.detailPanel.open(node);
          // Reciprocal sync: highlight matching citation card if active
          if (this.currentAnswerData?.citations) {
            setActiveCitationCard(this.dom.answerContainer, node.id);
          }
        }
      });
      this.graphRenderer.setData(this.graphData.nodes, this.graphData.edges);

      // 5. Initialize Reactive Filter Engine & List View
      this.filterEngine = new FilterEngine(this.graphData.nodes, this.graphData.edges, {
        onFilterChange: (filteredNodes, filteredEdges) => {
          this.graphRenderer.updateFilteredData(filteredNodes, filteredEdges);
        },
        onSelectNode: (node) => {
          this.graphRenderer.focusNode(node.id);
          this.detailPanel.open(node);
          if (this.currentAnswerData?.citations) {
            setActiveCitationCard(this.dom.answerContainer, node.id);
          }
        }
      });

      // Run initial filter to sync counts & list view
      this.filterEngine.applyFilters();

      // 6. Initialize Conversational Q&A Search Controller (Task 4.3)
      this.initQASearch();

      // 7. Dismiss Loading
      this.hideLoading();
    } catch (err) {
      console.error('[App] Startup failure:', err);
      this.showError(err.message);
    }
  }

  /**
   * Initializes the conversational Q&A search bar and citation-graph bridge.
   */
  initQASearch() {
    this.searchController = new SearchController({
      onLoading: (isLoading) => {
        if (isLoading) {
          clearAnswer(this.dom.answerContainer);
        }
      },
      onSuccess: (answerData, query) => {
        this.currentAnswerData = answerData;

        renderAnswer(this.dom.answerContainer, answerData, query, {
          onCitationClick: (citation, index, itemId) => {
            const node = findNodeForCitation(this.graphData.nodes, citation);
            const targetId = node ? node.id : itemId;
            focusCitation(this.graphRenderer, targetId, this.detailPanel, this.graphData.nodes);
          },
          onClearHighlights: () => {
            clearCitations(this.graphRenderer);
          },
          onClose: () => {
            clearCitations(this.graphRenderer);
            this.currentAnswerData = null;
          }
        });

        // Visually highlight all cited nodes and fit camera to the cited subgraph
        if (answerData.citations && answerData.citations.length > 0) {
          highlightCitations(
            this.graphRenderer,
            answerData.citations,
            this.graphData.nodes,
            this.graphData.edges
          );
        }
      },
      onError: () => {
        clearAnswer(this.dom.answerContainer);
        clearCitations(this.graphRenderer);
        this.currentAnswerData = null;
      },
      onClear: () => {
        clearAnswer(this.dom.answerContainer);
        clearCitations(this.graphRenderer);
        this.currentAnswerData = null;
      }
    });
  }

  updateHeaderStats({ nodes, edges }) {
    let projectsCount = 0;
    let areasCount = 0;
    let resourcesCount = 0;
    let archivesCount = 0;

    for (const node of nodes) {
      if (node.type === 'Item') {
        const cat = node.properties?.category || node.group;
        if (cat === 'Projects') projectsCount++;
        else if (cat === 'Areas') areasCount++;
        else if (cat === 'Resources') resourcesCount++;
        else if (cat === 'Archives') archivesCount++;
      }
    }

    if (this.dom.statTotal) this.dom.statTotal.textContent = nodes.length;
    if (this.dom.statProjects) this.dom.statProjects.textContent = projectsCount;
    if (this.dom.statAreas) this.dom.statAreas.textContent = areasCount;
    if (this.dom.statResources) this.dom.statResources.textContent = resourcesCount;
    if (this.dom.statArchives) this.dom.statArchives.textContent = archivesCount;
    if (this.dom.statEdges) this.dom.statEdges.textContent = edges.length;
  }

  showLoading(msg) {
    if (this.dom.loadingOverlay) {
      this.dom.loadingOverlay.classList.remove('hidden');
    }
    if (this.dom.loadingMsg) {
      this.dom.loadingMsg.textContent = msg;
    }
    if (this.dom.errorOverlay) {
      this.dom.errorOverlay.classList.add('hidden');
    }
  }

  hideLoading() {
    if (this.dom.loadingOverlay) {
      this.dom.loadingOverlay.classList.add('hidden');
    }
  }

  showError(msg) {
    if (this.dom.loadingOverlay) {
      this.dom.loadingOverlay.classList.add('hidden');
    }
    if (this.dom.errorOverlay) {
      this.dom.errorOverlay.classList.remove('hidden');
    }
    if (this.dom.errorMsg) {
      this.dom.errorMsg.textContent = msg;
    }
  }
}

// Global bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new SecondSelfApp();
  app.start();
});
