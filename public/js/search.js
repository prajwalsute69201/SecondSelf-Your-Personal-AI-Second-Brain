/**
 * SecondSelf — Q&A Search Controller (Task 4.3)
 *
 * Manages the conversational Q&A search bar:
 * - Accessible form submission and input validation
 * - Same-origin relative HTTP requests to /api/ask (Task 4.2 API contract)
 * - AbortController request lifecycle and 25-second timeout management
 * - Loading and error states with non-technical messages and retry capability
 * - Keyboard navigation (Enter to submit, Esc to clear, '/' to focus)
 */

/**
 * Validates the structure of the JSON response from /api/ask.
 *
 * @param {any} data Raw parsed JSON from API
 * @returns {boolean} True if payload satisfies the Task 4.2 Answer contract
 */
export function validateApiResponse(data) {
  if (!data || typeof data !== 'object') return false;
  if (data.ok !== true) return false;
  if (typeof data.answer !== 'string' || !data.answer) return false;
  if (!Array.isArray(data.citations)) return false;
  return true;
}

export class SearchController {
  /**
   * @param {Object} options Configuration callbacks and elements
   * @param {string} [options.endpoint] API endpoint URL (default: '/api/ask')
   * @param {Function} [options.onLoading] Called with boolean when request starts/ends
   * @param {Function} [options.onSuccess] Called with (answerData, query)
   * @param {Function} [options.onError] Called with (errorMessage, query)
   * @param {Function} [options.onClear] Called when query/highlights are cleared
   */
  constructor(options = {}) {
    this.endpoint = options.endpoint || '/api/ask';
    this.onLoading = options.onLoading || (() => {});
    this.onSuccess = options.onSuccess || (() => {});
    this.onError = options.onError || (() => {});
    this.onClear = options.onClear || (() => {});

    this.activeController = null;
    this.lastQuery = '';
    this.lastNoLLM = false;
    this.isLoading = false;

    this.initDOM();
    this.bindEvents();
  }

  initDOM() {
    this.form = document.getElementById('qa-search-form');
    this.input = document.getElementById('qa-search-input');
    this.submitBtn = document.getElementById('qa-submit-btn');
    this.clearBtn = document.getElementById('qa-clear-input-btn');
    this.noLLMToggle = document.getElementById('qa-no-llm-toggle');
    this.statusLive = document.getElementById('qa-status-live');
    this.errorBox = document.getElementById('qa-error-box');
    this.errorMsg = document.getElementById('qa-error-msg');
    this.retryBtn = document.getElementById('qa-retry-btn');

    // Read custom endpoint from data-api attribute if present
    if (this.form && this.form.dataset.api) {
      this.endpoint = this.form.dataset.api;
    }
  }

  bindEvents() {
    if (this.form) {
      this.form.addEventListener('submit', (e) => {
        e.preventDefault();
        this.submitQuery();
      });
    }

    if (this.input) {
      this.input.addEventListener('input', () => {
        const hasText = Boolean(this.input.value.trim());
        if (this.clearBtn) {
          this.clearBtn.classList.toggle('hidden', !hasText);
        }
      });

      this.input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          e.preventDefault();
          if (this.input.value) {
            this.clearInput();
          } else {
            this.onClear();
          }
        }
      });
    }

    if (this.clearBtn) {
      this.clearBtn.addEventListener('click', () => this.clearInput());
    }

    if (this.retryBtn) {
      this.retryBtn.addEventListener('click', () => {
        if (this.lastQuery) {
          this.submitQuery(this.lastQuery, this.lastNoLLM);
        }
      });
    }

    // Global keyboard shortcut: Ctrl+K / Cmd+K to focus Q&A search input
    window.addEventListener('keydown', (e) => {
      const isCmdOrCtrlK = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k';
      if (isCmdOrCtrlK) {
        e.preventDefault();
        this.input?.focus();
        this.input?.select();
      }
    });
  }

  /**
   * Submits a question to the serverless synthesis endpoint.
   *
   * @param {string} [overrideQuery] Optional query override (used by retry)
   * @param {boolean} [overrideNoLLM] Optional mode override
   */
  async submitQuery(overrideQuery = null, overrideNoLLM = null) {
    const query = (overrideQuery ?? this.input?.value ?? '').trim();
    const noLLM = overrideNoLLM ?? (this.noLLMToggle?.checked ?? false);

    // Empty-question guard
    if (!query) {
      this.input?.focus();
      this.announceStatus('Please enter a question to ask SecondSelf.');
      return;
    }

    // Length boundary guard (<= 1000 characters per Task 4.2 contract)
    if (query.length > 1000) {
      this.showError('Question exceeds maximum allowed length of 1000 characters.');
      return;
    }

    // Abort previous in-flight request if user submits again
    if (this.activeController) {
      this.activeController.abort();
    }

    this.activeController = new AbortController();
    this.lastQuery = query;
    this.lastNoLLM = noLLM;

    this.setLoading(true);
    this.hideError();
    this.announceStatus(`Searching knowledge corpus for: "${query}"...`);

    const timeoutId = setTimeout(() => {
      if (this.activeController) {
        this.activeController.abort();
      }
    }, 25000); // 25s timeout ceiling

    try {
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          question: query,
          top_k: 5,
          min_score: 0.35,
          no_llm: noLLM,
          expand_graph: false
        }),
        signal: this.activeController.signal
      });

      clearTimeout(timeoutId);

      let payload;
      try {
        payload = await response.json();
      } catch (parseErr) {
        throw new Error(`The question engine returned an invalid response (HTTP ${response.status}).`);
      }

      if (!response.ok || !validateApiResponse(payload)) {
        const errorDetail = payload?.error || `Request failed with HTTP ${response.status}`;
        throw new Error(errorDetail);
      }

      // Success
      this.announceStatus(
        `Answer synthesized with ${payload.citations.length} cited sources.`
      );
      this.onSuccess(payload, query);
    } catch (err) {
      clearTimeout(timeoutId);

      if (err.name === 'AbortError') {
        console.warn('[Search] Request aborted or timed out');
        this.showError('Search request timed out. Please check your connection and retry.');
        this.onError('Search request timed out.', query);
      } else {
        console.error('[Search] Q&A API error:', err);
        const friendlyMessage = this.formatErrorMessage(err.message);
        this.showError(friendlyMessage);
        this.onError(friendlyMessage, query);
      }
    } finally {
      this.setLoading(false);
      this.activeController = null;
    }
  }

  /**
   * Translates technical error messages into clear, non-technical explanations.
   *
   * @param {string} rawMessage
   * @returns {string}
   */
  formatErrorMessage(rawMessage) {
    const lower = rawMessage.toLowerCase();
    if (lower.includes('failed to fetch') || lower.includes('networkerror')) {
      return 'Unable to reach the SecondSelf server. Please check your internet connection or verify the backend is online.';
    }
    if (lower.includes('timed out')) {
      return 'The request took too long to complete. Please try asking again.';
    }
    if (lower.includes('missing') && lower.includes('groq_api_key')) {
      return 'Groq API Key is not configured on the server. You can still query in Retrieval-Only mode.';
    }
    return rawMessage || 'An unexpected error occurred while synthesizing your answer.';
  }

  setLoading(isLoading) {
    this.isLoading = isLoading;
    if (this.submitBtn) {
      this.submitBtn.disabled = isLoading;
      const spinner = this.submitBtn.querySelector('.btn-spinner');
      const text = this.submitBtn.querySelector('.btn-text');
      spinner?.classList.toggle('hidden', !isLoading);
      if (text) {
        text.textContent = isLoading ? 'Synthesizing...' : 'Ask AI';
      }
    }
    if (this.input) {
      this.input.disabled = isLoading;
    }
    this.onLoading(isLoading);
  }

  showError(message) {
    if (this.errorBox && this.errorMsg) {
      this.errorMsg.textContent = message;
      this.errorBox.classList.remove('hidden');
      this.announceStatus(`Error: ${message}`);
    }
  }

  hideError() {
    this.errorBox?.classList.add('hidden');
  }

  clearInput() {
    if (this.input) {
      this.input.value = '';
      this.input.focus();
    }
    this.clearBtn?.classList.add('hidden');
    this.hideError();
  }

  announceStatus(message) {
    if (this.statusLive) {
      this.statusLive.textContent = message;
    }
  }
}
