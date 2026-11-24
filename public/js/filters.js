/**
 * SecondSelf — Filter Engine & List Explorer (Task 3.2)
 *
 * Manages reactive filter state (Search, PARA Categories, Confidence threshold),
 * coordinates filtered sub-graphs for Vis.js/Canvas, and renders the accessible List View.
 */

export class FilterEngine {
  constructor(allNodes, allEdges, options = {}) {
    this.allNodes = allNodes;
    this.allEdges = allEdges;
    this.onFilterChange = options.onFilterChange || (() => {});
    this.onSelectNode = options.onSelectNode || (() => {});

    // Default filter state
    this.state = {
      searchQuery: '',
      selectedCategories: new Set(['Projects', 'Areas', 'Resources', 'Archives']),
      minConfidence: 0.0,
      selectedType: 'ALL',
      viewMode: 'graph' // 'graph' or 'list'
    };

    // Debounce timer for search
    this.debounceTimer = null;

    this.initDOMElements();
    this.bindEvents();
  }

  initDOMElements() {
    this.dom = {
      searchInput: document.getElementById('search-input'),
      searchClearBtn: document.getElementById('search-clear-btn'),
      catPills: document.querySelectorAll('.cat-pill'),
      confidenceSlider: document.getElementById('confidence-slider'),
      confidenceVal: document.getElementById('confidence-val'),
      resultsCounter: document.getElementById('results-counter'),
      resetFiltersBtn: document.getElementById('reset-filters-btn'),
      listView: document.getElementById('list-view'),
      listGrid: document.getElementById('list-grid'),
      graphView: document.getElementById('graph-view'),
      viewGraphBtn: document.getElementById('view-graph-btn'),
      viewListBtn: document.getElementById('view-list-btn')
    };
  }

  bindEvents() {
    // 1. Search Input (Debounced)
    if (this.dom.searchInput) {
      this.dom.searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        if (this.dom.searchClearBtn) {
          this.dom.searchClearBtn.style.display = query ? 'block' : 'none';
        }
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
          this.state.searchQuery = query.toLowerCase();
          this.applyFilters();
        }, 150);
      });
    }

    if (this.dom.searchClearBtn) {
      this.dom.searchClearBtn.addEventListener('click', () => {
        if (this.dom.searchInput) {
          this.dom.searchInput.value = '';
          this.dom.searchInput.focus();
        }
        this.dom.searchClearBtn.style.display = 'none';
        this.state.searchQuery = '';
        this.applyFilters();
      });
    }

    // 2. Category Filter Pills
    if (this.dom.catPills) {
      this.dom.catPills.forEach((pill) => {
        pill.addEventListener('click', () => {
          const category = pill.dataset.category;
          if (!category) return;

          const isPressed = pill.getAttribute('aria-pressed') === 'true';
          const nextState = !isPressed;
          pill.setAttribute('aria-pressed', String(nextState));

          if (nextState) {
            this.state.selectedCategories.add(category);
          } else {
            this.state.selectedCategories.delete(category);
          }

          this.applyFilters();
        });
      });
    }

    // 3. Confidence Slider
    if (this.dom.confidenceSlider) {
      this.dom.confidenceSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        this.state.minConfidence = val;
        if (this.dom.confidenceVal) {
          this.dom.confidenceVal.textContent = val.toFixed(2);
        }
        this.applyFilters();
      });
    }

    // 4. Reset Filters Button
    if (this.dom.resetFiltersBtn) {
      this.dom.resetFiltersBtn.addEventListener('click', () => {
        this.resetFilters();
      });
    }

    // 5. View Mode Switcher
    if (this.dom.viewGraphBtn && this.dom.viewListBtn) {
      this.dom.viewGraphBtn.addEventListener('click', () => this.setViewMode('graph'));
      this.dom.viewListBtn.addEventListener('click', () => this.setViewMode('list'));
    }

    // 6. Global keyboard shortcuts (/ to search, Esc to clear)
    window.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== this.dom.searchInput) {
        e.preventDefault();
        this.dom.searchInput?.focus();
      } else if (e.key === 'Escape' && document.activeElement === this.dom.searchInput) {
        if (this.dom.searchInput.value) {
          this.dom.searchInput.value = '';
          this.state.searchQuery = '';
          this.dom.searchClearBtn.style.display = 'none';
          this.applyFilters();
        }
        this.dom.searchInput.blur();
      }
    });
  }

  /**
   * Resets all filters to initial default state.
   */
  resetFilters() {
    this.state.searchQuery = '';
    this.state.selectedCategories = new Set(['Projects', 'Areas', 'Resources', 'Archives']);
    this.state.minConfidence = 0.0;

    if (this.dom.searchInput) {
      this.dom.searchInput.value = '';
    }
    if (this.dom.searchClearBtn) {
      this.dom.searchClearBtn.style.display = 'none';
    }
    if (this.dom.catPills) {
      this.dom.catPills.forEach((p) => p.setAttribute('aria-pressed', 'true'));
    }
    if (this.dom.confidenceSlider) {
      this.dom.confidenceSlider.value = '0.0';
    }
    if (this.dom.confidenceVal) {
      this.dom.confidenceVal.textContent = '0.00';
    }

    this.applyFilters();
  }

  /**
   * Switches view between 'graph' and 'list'.
   */
  setViewMode(mode) {
    this.state.viewMode = mode;

    if (mode === 'graph') {
      this.dom.viewGraphBtn?.classList.add('active');
      this.dom.viewGraphBtn?.setAttribute('aria-selected', 'true');
      this.dom.viewListBtn?.classList.remove('active');
      this.dom.viewListBtn?.setAttribute('aria-selected', 'false');

      this.dom.graphView?.classList.remove('hidden');
      this.dom.listView?.classList.add('hidden');
    } else {
      this.dom.viewListBtn?.classList.add('active');
      this.dom.viewListBtn?.setAttribute('aria-selected', 'true');
      this.dom.viewGraphBtn?.classList.remove('active');
      this.dom.viewGraphBtn?.setAttribute('aria-selected', 'false');

      this.dom.listView?.classList.remove('hidden');
      this.dom.graphView?.classList.add('hidden');
    }
  }

  /**
   * Executes filter evaluation and notifies subscribers.
   */
  applyFilters() {
    const { searchQuery, selectedCategories, minConfidence } = this.state;

    // Filter nodes
    const filteredNodes = this.allNodes.filter((node) => {
      // 1. PARA Category matching
      if (node.type === 'Item') {
        const cat = node.properties.category || node.group;
        if (!selectedCategories.has(cat)) {
          return false;
        }

        // Confidence threshold
        const conf = node.properties.confidence;
        if (typeof conf === 'number' && conf < minConfidence) {
          return false;
        }
      } else if (node.type === 'PARA_Category') {
        // e.g. para:projects -> Projects
        const catName = node.label;
        if (!selectedCategories.has(catName)) {
          return false;
        }
      }

      // 2. Search query matching
      if (searchQuery) {
        const label = (node.label || '').toLowerCase();
        const summary = (node.summary || '').toLowerCase();
        const id = (node.id || '').toLowerCase();
        const tags = (node.tags || []).join(' ').toLowerCase();
        const rationale = (node.properties?.rationale || '').toLowerCase();

        const match = label.includes(searchQuery) ||
                      summary.includes(searchQuery) ||
                      tags.includes(searchQuery) ||
                      rationale.includes(searchQuery) ||
                      id.includes(searchQuery);
        if (!match) return false;
      }

      return true;
    });

    const visibleNodeIdSet = new Set(filteredNodes.map((n) => n.id));

    // Filter edges: both endpoints must be visible
    const filteredEdges = this.allEdges.filter((edge) => {
      const from = edge.from || edge.source;
      const to = edge.to || edge.target;
      return visibleNodeIdSet.has(from) && visibleNodeIdSet.has(to);
    });

    // Update results counter text
    this.updateResultsCounter(filteredNodes.length, this.allNodes.length);

    // Render list view cards
    this.renderListView(filteredNodes);

    // Notify graph and application orchestrator
    this.onFilterChange(filteredNodes, filteredEdges, visibleNodeIdSet);
  }

  /**
   * Updates the accessible counter element with ARIA live region.
   */
  updateResultsCounter(visibleCount, totalCount) {
    if (!this.dom.resultsCounter) return;

    if (visibleCount === totalCount) {
      this.dom.resultsCounter.innerHTML = `Showing all <strong>${totalCount}</strong> nodes`;
    } else {
      this.dom.resultsCounter.innerHTML = `Showing <strong>${visibleCount}</strong> of ${totalCount} nodes`;
    }
  }

  /**
   * Renders item cards for the List/Card view.
   */
  renderListView(nodes) {
    if (!this.dom.listGrid) return;

    // Filter to primarily Items for clean browsing (or include all)
    const items = nodes.filter((n) => n.type === 'Item');

    if (items.length === 0) {
      this.dom.listGrid.innerHTML = `
        <div class="empty-filter-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <h3>No matching notes found</h3>
          <p>Try loosening your search query, selecting more categories, or lowering the confidence threshold.</p>
          <button type="button" class="retry-btn" id="empty-reset-btn">Reset All Filters</button>
        </div>
      `;
      document.getElementById('empty-reset-btn')?.addEventListener('click', () => this.resetFilters());
      return;
    }

    this.dom.listGrid.innerHTML = items.map((item) => {
      const cat = item.properties.category || item.group || 'Resource';
      const catLower = cat.toLowerCase().replace(/s$/, ''); // project, area, resource, archive
      const conf = typeof item.properties.confidence === 'number'
        ? Math.round(item.properties.confidence * 100)
        : null;
      const confBarWidth = conf !== null ? conf : 0;
      const tags = (item.tags || []).slice(0, 3);

      return `
        <article class="item-card" data-node-id="${item.id}" tabindex="0" role="button" aria-label="View note: ${this.escapeHtml(item.label)}">
          <div class="card-header">
            <h3 class="card-title">${this.escapeHtml(item.label)}</h3>
            <span class="card-badge cat-pill cat-${catLower}" aria-pressed="true">${cat}</span>
          </div>

          <p class="card-snippet">${this.escapeHtml(item.summary || item.label)}</p>

          ${conf !== null ? `
            <div class="card-confidence-bar">
              <span>Confidence</span>
              <div class="bar-track">
                <div class="bar-fill" style="width: ${confBarWidth}%"></div>
              </div>
              <span>${conf}%</span>
            </div>
          ` : ''}

          <div class="card-footer">
            <div class="card-tags">
              ${tags.map((t) => `<span class="tag-badge">#${this.escapeHtml(t)}</span>`).join('')}
              ${item.tags && item.tags.length > 3 ? `<span class="tag-badge">+${item.tags.length - 3}</span>` : ''}
            </div>
            <span>${item.degree || 0} links</span>
          </div>
        </article>
      `;
    }).join('');

    // Attach click and Enter listeners to cards
    const cards = this.dom.listGrid.querySelectorAll('.item-card');
    cards.forEach((card) => {
      const nodeId = card.dataset.nodeId;
      const targetNode = this.allNodes.find((n) => n.id === nodeId);

      const select = () => {
        if (targetNode) {
          this.onSelectNode(targetNode);
        }
      };

      card.addEventListener('click', select);
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      });
    });
  }

  escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}
