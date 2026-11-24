/**
 * SecondSelf — Slide-out Detail Drawer (Task 3.2)
 *
 * Accessible drawer component displaying comprehensive note metadata,
 * PARA category decision rationale, confidence scores, provenance, and interactive backlinks.
 * Conforms to WCAG 2.1 AA dialog/drawer accessibility guidelines.
 */

export class DetailPanel {
  constructor(allNodes, allEdges, options = {}) {
    this.allNodes = allNodes;
    this.allEdges = allEdges;
    this.onSelectNode = options.onSelectNode || (() => {});
    this.onFilterByTag = options.onFilterByTag || (() => {});

    this.currentNode = null;
    this.previousActiveElement = null;

    this.initDOMElements();
    this.bindEvents();
  }

  initDOMElements() {
    this.dom = {
      drawer: document.getElementById('detail-drawer'),
      backdrop: document.getElementById('drawer-backdrop'),
      closeBtn: document.getElementById('drawer-close-btn'),
      typeLabel: document.getElementById('drawer-type-label'),
      title: document.getElementById('drawer-title'),
      body: document.getElementById('drawer-body')
    };
  }

  bindEvents() {
    if (this.dom.closeBtn) {
      this.dom.closeBtn.addEventListener('click', () => this.close());
    }

    if (this.dom.backdrop) {
      this.dom.backdrop.addEventListener('click', () => this.close());
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) {
        e.preventDefault();
        this.close();
      }
    });
  }

  isOpen() {
    return this.dom.drawer?.classList.contains('open') ?? false;
  }

  /**
   * Opens the drawer and renders metadata for the target node.
   *
   * @param {Object} node Target node object
   */
  open(node) {
    if (!node) return;
    this.currentNode = node;
    this.previousActiveElement = document.activeElement;

    // Update Drawer Header
    if (this.dom.typeLabel) {
      this.dom.typeLabel.textContent = node.type || 'Item';
    }
    if (this.dom.title) {
      this.dom.title.textContent = node.label || 'Untitled';
    }

    // Render Body Sections
    this.renderBody(node);

    // Show Drawer with Animation & Accessibility attributes
    this.dom.drawer?.classList.add('open');
    this.dom.drawer?.setAttribute('aria-hidden', 'false');
    this.dom.backdrop?.classList.add('open');

    // Move focus to close button
    setTimeout(() => {
      this.dom.closeBtn?.focus();
    }, 100);
  }

  /**
   * Closes the drawer and restores focus.
   */
  close() {
    if (!this.isOpen()) return;

    this.dom.drawer?.classList.remove('open');
    this.dom.drawer?.setAttribute('aria-hidden', 'true');
    this.dom.backdrop?.classList.remove('open');
    this.currentNode = null;

    // Restore focus to previously active element
    if (this.previousActiveElement && typeof this.previousActiveElement.focus === 'function') {
      this.previousActiveElement.focus();
    }
  }

  /**
   * Renders the complete drawer body content.
   */
  renderBody(node) {
    if (!this.dom.body) return;

    const props = node.properties || {};
    const cat = props.category || node.group || 'Resource';
    const catLower = cat.toLowerCase().replace(/s$/, '');
    const conf = typeof props.confidence === 'number' ? Math.round(props.confidence * 100) : null;
    const rationale = props.rationale || this.generateDefaultRationale(node);
    const tags = Array.isArray(node.tags) ? node.tags : (props.tags || []);
    const summary = node.summary || props.summary || 'No summary text available.';
    const backlinks = this.findConnectedNodes(node.id);

    this.dom.body.innerHTML = `
      <!-- 1. Category & Confidence -->
      <section class="drawer-section" aria-label="PARA Classification">
        <h4 class="drawer-section-title">Classification &amp; Confidence</h4>
        <div class="para-classification-card">
          <div class="classification-header">
            <span class="card-badge cat-pill cat-${catLower}" aria-pressed="true">
              ${cat}
            </span>
            ${props.needs_review ? `
              <span class="card-badge" style="background: var(--color-danger-bg); color: var(--color-danger); border: 1px solid var(--color-danger);">
                Needs Review
              </span>
            ` : ''}
          </div>

          ${conf !== null ? `
            <div class="confidence-meter">
              <span class="confidence-val-lg">${conf}%</span>
              <div class="bar-track" style="height: 7px;">
                <div class="bar-fill" style="width: ${conf}%; background: ${conf >= 70 ? 'var(--color-resource)' : (conf >= 40 ? 'var(--color-project)' : 'var(--color-danger)')};"></div>
              </div>
            </div>
          ` : ''}

          ${rationale ? `
            <blockquote class="rationale-quote" title="Autonomous Classification Rationale">
              <strong>Rationale:</strong> ${this.escapeHtml(rationale)}
            </blockquote>
          ` : ''}
        </div>
      </section>

      <!-- 2. Content / Summary Preview -->
      <section class="drawer-section" aria-label="Content Summary">
        <h4 class="drawer-section-title">Summary</h4>
        <p style="font-size: 0.88rem; line-height: 1.55; color: var(--text-secondary); white-space: pre-wrap;">
          ${this.escapeHtml(summary)}
        </p>
      </section>

      <!-- 3. Tags -->
      ${tags.length > 0 ? `
        <section class="drawer-section" aria-label="Tags">
          <h4 class="drawer-section-title">Tags</h4>
          <div class="card-tags">
            ${tags.map((t) => `
              <button type="button" class="tag-badge btn-tag" data-tag="${this.escapeHtml(t)}" title="Filter by #${this.escapeHtml(t)}">
                #${this.escapeHtml(t)}
              </button>
            `).join('')}
          </div>
        </section>
      ` : ''}

      <!-- 4. Connected Relationships / Backlinks -->
      <section class="drawer-section" aria-label="Relationships">
        <h4 class="drawer-section-title">
          Connected Relationships (${backlinks.length})
        </h4>
        ${backlinks.length === 0 ? `
          <p style="font-size: 0.82rem; color: var(--text-dim); font-style: italic;">
            No direct connections found in graph.
          </p>
        ` : `
          <div class="backlinks-list">
            ${backlinks.map((link) => `
              <button type="button" class="backlink-item" data-target-id="${link.peer.id}">
                <div class="backlink-title" title="${this.escapeHtml(link.peer.label)}">
                  ${this.escapeHtml(link.peer.label)}
                </div>
                <div style="display: flex; align-items: center; gap: 0.35rem;">
                  ${link.similarity ? `<span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--color-item);">${Math.round(link.similarity * 100)}%</span>` : ''}
                  <span class="backlink-badge">${link.type}</span>
                </div>
              </button>
            `).join('')}
          </div>
        `}
      </section>

      <!-- 5. Provenance & Technical Metadata -->
      <section class="drawer-section" aria-label="Metadata Provenance">
        <h4 class="drawer-section-title">Provenance</h4>
        <div class="meta-grid">
          <div class="meta-field">
            <span class="meta-key">Record ID</span>
            <span class="meta-val" title="${props.record_id || node.id}">${props.record_id || node.id}</span>
          </div>
          <div class="meta-field">
            <span class="meta-key">Source Type</span>
            <span class="meta-val">${props.source_type || 'markdown'}</span>
          </div>
          <div class="meta-field">
            <span class="meta-key">Created</span>
            <span class="meta-val">${this.formatDate(props.created_at)}</span>
          </div>
          <div class="meta-field">
            <span class="meta-key">Word Count</span>
            <span class="meta-val">${props.word_count || 0}</span>
          </div>
          ${props.file_path ? `
            <div class="meta-field" style="grid-column: 1 / -1;">
              <span class="meta-key">File Path</span>
              <span class="meta-val" title="${this.escapeHtml(props.file_path)}">${this.escapeHtml(props.file_path)}</span>
            </div>
          ` : ''}
          ${props.content_hash ? `
            <div class="meta-field" style="grid-column: 1 / -1;">
              <span class="meta-key">SHA-256 Hash</span>
              <span class="meta-val" title="${props.content_hash}">${props.content_hash.slice(0, 16)}...</span>
            </div>
          ` : ''}
        </div>
      </section>
    `;

    // Wire up tag pill clicks
    const tagBtns = this.dom.body.querySelectorAll('.btn-tag');
    tagBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const tag = btn.dataset.tag;
        if (tag) {
          this.onFilterByTag(tag);
        }
      });
    });

    // Wire up backlink clicks
    const backlinkBtns = this.dom.body.querySelectorAll('.backlink-item');
    backlinkBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetId = btn.dataset.targetId;
        const targetNode = this.allNodes.find((n) => n.id === targetId);
        if (targetNode) {
          this.onSelectNode(targetNode);
        }
      });
    });
  }

  /**
   * Finds all nodes connected to the given node ID with their edge metadata.
   */
  findConnectedNodes(nodeId) {
    const results = [];
    const nodeMap = new Map(this.allNodes.map((n) => [n.id, n]));

    for (const edge of this.allEdges) {
      const from = edge.from || edge.source;
      const to = edge.to || edge.target;

      if (from === nodeId) {
        const peer = nodeMap.get(to);
        if (peer) {
          results.push({
            peer,
            type: edge.type || 'LINK',
            similarity: edge.properties?.similarity_score || null
          });
        }
      } else if (to === nodeId) {
        const peer = nodeMap.get(from);
        if (peer) {
          results.push({
            peer,
            type: edge.type || 'LINK',
            similarity: edge.properties?.similarity_score || null
          });
        }
      }
    }

    return results;
  }

  generateDefaultRationale(node) {
    if (node.type === 'PARA_Category') {
      return `Core PARA taxonomy category container for ${node.label}.`;
    }
    if (node.type === 'Tag') {
      return `Topic taxonomy clustering notes tagged with #${node.label}.`;
    }
    if (node.type === 'Source') {
      return `Origin adapter provenance tracking items extracted via ${node.label}.`;
    }
    return 'Autonomous classification assigned based on content semantics and keyword rules.';
  }

  formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    try {
      const d = new Date(dateStr);
      return isNaN(d.getTime()) ? dateStr : d.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return dateStr;
    }
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
