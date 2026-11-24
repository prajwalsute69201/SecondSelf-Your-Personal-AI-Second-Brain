/**
 * SecondSelf — Citation & Answer Renderer (Task 4.3)
 *
 * Renders the synthesized Q&A answer:
 * - Formats answer text with clickable, accessible inline citation superscripts [1], [2]
 * - Displays source citation cards with similarity scores, snippets, and PARA category tags
 * - Highlights active citations and supports keyboard navigation
 * - Clearly indicates retrieval-only mode and "not found in corpus" states
 */

/**
 * Escapes HTML characters to prevent XSS vulnerabilities.
 *
 * @param {string} str Raw untrusted string
 * @returns {string} Sanitized HTML safe string
 */
export function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Formats markdown-like emphasis (**bold**, `code`) and transforms
 * citations into interactive, accessible button elements.
 *
 * @param {string} rawText Raw synthesized answer text
 * @param {Array<Object>} citations Array of citation objects
 * @returns {string} Render-ready HTML string
 */
export function formatAnswerText(rawText, citations = []) {
  if (!rawText) return '';

  let html = escapeHtml(rawText);

  // 1. Basic formatting: bold and inline code
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // 2. Replace numeric inline citations like [1], [2], [3]
  html = html.replace(/\[(\d+)\]/g, (match, digits) => {
    const idx = parseInt(digits, 10);
    const citation = citations[idx - 1];
    const title = citation ? escapeHtml(citation.title) : `Source ${idx}`;
    const itemId = citation ? escapeHtml(citation.item_id) : '';
    return `<button type="button" class="inline-citation" data-citation-index="${idx}" data-item-id="${itemId}" aria-label="Jump to Citation ${idx}: ${title}" title="Inspect ${title}">[${idx}]</button>`;
  });

  // 3. Replace item citations like [item:note-1] or [note-1]
  citations.forEach((c, idx) => {
    const citationNum = idx + 1;
    const cleanId = c.item_id.replace(/^item:/, '');
    const pattern1 = new RegExp(`\\[${escapeRegex(c.item_id)}\\]`, 'g');
    const pattern2 = new RegExp(`\\[${escapeRegex(cleanId)}\\]`, 'g');
    const btn = `<button type="button" class="inline-citation" data-citation-index="${citationNum}" data-item-id="${escapeHtml(c.item_id)}" aria-label="Jump to Citation ${citationNum}: ${escapeHtml(c.title)}">[${citationNum}]</button>`;

    html = html.replace(pattern1, btn);
    html = html.replace(pattern2, btn);
  });

  // 4. Preserve paragraph breaks
  const paragraphs = html.split(/\n\s*\n/).filter(Boolean);
  return paragraphs.map((p) => `<p class="qa-answer-para">${p.replace(/\n/g, '<br>')}</p>`).join('');
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Checks if the answer text indicates a "not found in corpus" result.
 *
 * @param {string} text
 * @param {Array} citations
 * @returns {boolean}
 */
export function isNotFoundAnswer(text, citations) {
  if (!text) return true;
  if (!citations || citations.length === 0) return true;
  const lower = text.toLowerCase();
  return (
    lower.includes('not_found_in_corpus') ||
    lower.includes('could not find any relevant information') ||
    lower.includes('no relevant items found in your corpus')
  );
}

/**
 * Renders the synthesized answer and citation cards into the answer container.
 *
 * @param {HTMLElement} container The DOM element to render into
 * @param {Object} answerData Structured payload returned by /api/ask
 * @param {string} query The user's query
 * @param {Object} options Callbacks and actions
 * @param {Function} [options.onCitationClick] Called when a citation is clicked: (citation, index, nodeId)
 * @param {Function} [options.onClearHighlights] Called when "Clear Highlights" is clicked
 * @param {Function} [options.onClose] Called when answer panel is dismissed
 */
export function renderAnswer(container, answerData, query, options = {}) {
  if (!container || !answerData) return;

  const {
    answer = '',
    citations = [],
    mode = 'llm_synthesis',
    model = null,
    latency_ms = 0,
    is_grounded = true,
    uncited_claims_flag = false
  } = answerData;

  const notFound = isNotFoundAnswer(answer, citations);
  const formattedHtml = formatAnswerText(answer, citations);
  const isRetrievalOnly = mode === 'retrieval_only';

  container.innerHTML = `
    <article class="qa-answer-card" role="region" aria-label="Synthesized answer for ${escapeHtml(query)}">
      <!-- 1. Header Toolbar -->
      <header class="qa-answer-header">
        <div class="qa-meta-group">
          <span class="qa-query-label">Question:</span>
          <span class="qa-query-text" title="${escapeHtml(query)}">"${escapeHtml(query)}"</span>
        </div>

        <div class="qa-badges-strip">
          ${
            isRetrievalOnly
              ? `<span class="qa-badge qa-badge-retrieval" title="Retrieved via dense semantic vector search without LLM synthesis">📚 Best Matches (Retrieval Only)</span>`
              : `<span class="qa-badge qa-badge-model" title="Synthesized with ${escapeHtml(model || 'LLM')}">✨ ${escapeHtml(model || 'AI Synthesizer')}</span>`
          }
          <span class="qa-badge qa-badge-grounded ${is_grounded ? 'grounded-pass' : 'grounded-warn'}" title="${is_grounded ? 'All claims cited from verified corpus notes' : 'Potential uncited claims detected'}">
            ${is_grounded ? '✓ Grounded' : '⚠️ Review Citations'}
          </span>
          <span class="qa-badge qa-badge-latency" title="Total pipeline latency">
            ⏱️ ${Math.round(latency_ms)}ms
          </span>
        </div>

        <div class="qa-header-actions">
          <button type="button" id="qa-clear-highlights-btn" class="qa-action-btn qa-clear-btn" aria-label="Clear active graph citation highlights" title="Clear Graph Highlights">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12"></path>
            </svg>
            Clear Highlights
          </button>
          <button type="button" id="qa-close-answer-btn" class="qa-action-btn qa-close-btn" aria-label="Close answer panel" title="Close Panel (Esc)">
            &times;
          </button>
        </div>
      </header>

      <!-- 2. Answer Body / Not Found Notice -->
      <div class="qa-answer-body">
        ${
          notFound
            ? `
            <div class="qa-not-found-banner" role="status">
              <div class="qa-not-found-icon" aria-hidden="true">🔍</div>
              <div class="qa-not-found-text">
                <h3>No Matching Notes Found in Corpus</h3>
                <p>We searched your personal notes and semantic embeddings, but could not find information addressing this query above the similarity threshold.</p>
                <p class="qa-not-found-hint">Tip: Try searching with broader keywords, note titles, or check the PARA categories in the graph below.</p>
              </div>
            </div>
          `
            : `
            <div class="qa-synthesized-text">
              ${formattedHtml}
            </div>
          `
        }

        ${
          uncited_claims_flag
            ? `<div class="qa-warning-banner" role="alert">
                <strong>Heuristic Notice:</strong> The synthesized answer may contain supplementary context not directly mapped to a specific citation. Please review the sources below.
               </div>`
            : ''
        }
      </div>

      <!-- 3. Citation Cards Section -->
      ${
        citations.length > 0
          ? `
          <footer class="qa-citations-footer">
            <div class="qa-citations-title-bar">
              <h3 class="qa-citations-heading">
                Cited Sources (${citations.length})
              </h3>
              <span class="qa-citations-subtext">Click any card or inline badge to focus the note in your knowledge graph</span>
            </div>

            <div class="qa-citation-grid" role="list" aria-label="Source citations">
              ${citations
                .map((c, i) => {
                  const idx = i + 1;
                  const matchPercent = Math.round((c.score || 0) * 100);
                  const cleanId = c.item_id ? c.item_id.replace(/^item:/, '') : '';
                  const sourceFile = c.source ? c.source.split('/').pop() : '';

                  return `
                  <div
                    class="citation-card"
                    role="listitem"
                    tabindex="0"
                    data-citation-index="${idx}"
                    data-item-id="${escapeHtml(c.item_id)}"
                    data-record-id="${escapeHtml(cleanId)}"
                    aria-label="Citation ${idx}: ${escapeHtml(c.title)}, ${matchPercent}% match"
                  >
                    <div class="citation-card-header">
                      <span class="citation-number-chip">[${idx}]</span>
                      <h4 class="citation-title" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</h4>
                      <span class="citation-score-badge" title="Cosine similarity match score">${matchPercent}% match</span>
                    </div>

                    ${
                      c.snippet
                        ? `<blockquote class="citation-snippet" title="Extracted excerpt">"${escapeHtml(c.snippet)}"</blockquote>`
                        : ''
                    }

                    <div class="citation-card-footer">
                      ${sourceFile ? `<span class="citation-source-path" title="${escapeHtml(c.source)}">📄 ${escapeHtml(sourceFile)}</span>` : ''}
                      <span class="citation-focus-hint" aria-hidden="true">Inspect node →</span>
                    </div>
                  </div>
                `;
                })
                .join('')}
            </div>
          </footer>
        `
          : ''
      }
    </article>
  `;

  // Reveal container
  container.classList.remove('hidden');

  // Bind inline citations
  container.querySelectorAll('.inline-citation').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.citationIndex, 10);
      const itemId = btn.dataset.itemId;
      const citation = citations[idx - 1];
      setActiveCitationCard(container, itemId, idx);
      options.onCitationClick?.(citation, idx, itemId);
    });
  });

  // Bind citation cards (click + keyboard Enter/Space)
  container.querySelectorAll('.citation-card').forEach((card) => {
    const handleSelect = () => {
      const idx = parseInt(card.dataset.citationIndex, 10);
      const itemId = card.dataset.itemId;
      const citation = citations[idx - 1];
      setActiveCitationCard(container, itemId, idx);
      options.onCitationClick?.(citation, idx, itemId);
    };

    card.addEventListener('click', handleSelect);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleSelect();
      }
    });
  });

  // Bind header buttons
  const clearBtn = container.querySelector('#qa-clear-highlights-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      container.querySelectorAll('.citation-card.active').forEach((c) => c.classList.remove('active'));
      options.onClearHighlights?.();
    });
  }

  const closeBtn = container.querySelector('#qa-close-answer-btn');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      clearAnswer(container);
      options.onClose?.();
    });
  }
}

/**
 * Sets the active visual state on a citation card and scrolls it into view.
 *
 * @param {HTMLElement} container Answer panel container
 * @param {string} itemId Target item/node ID
 * @param {number} [citationIndex] Optional 1-based index
 */
export function setActiveCitationCard(container, itemId, citationIndex = null) {
  if (!container) return;

  const cards = container.querySelectorAll('.citation-card');
  let targetCard = null;

  cards.forEach((card) => {
    const cardItemId = card.dataset.itemId;
    const cardRecId = card.dataset.recordId;
    const cardIdx = parseInt(card.dataset.citationIndex, 10);

    const isMatch =
      (itemId && (cardItemId === itemId || cardRecId === itemId || cardItemId?.replace(/^item:/, '') === itemId.replace(/^item:/, ''))) ||
      (citationIndex && cardIdx === citationIndex);

    if (isMatch) {
      card.classList.add('active');
      targetCard = card;
    } else {
      card.classList.remove('active');
    }
  });

  if (targetCard) {
    targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
  }
}

/**
 * Hides and empties the answer container.
 *
 * @param {HTMLElement} container
 */
export function clearAnswer(container) {
  if (container) {
    container.classList.add('hidden');
    container.innerHTML = '';
  }
}
