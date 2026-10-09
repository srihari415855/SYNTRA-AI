/**
 * SYNTRA AI — OpenPilot Engine
 * Frontend Orchestration, Authentication, Multimodal Ingress, and Telemetry Engine.
 */

(function () {
  'use strict';

  // Application State
  const state = {
    user: null,
    token: localStorage.getItem('syntra_token') || null,
    currentMode: 'best_value',
    activeTab: 'studio',
    imageAttachment: null, // { base64: string, name: string, size: string }
    isRecordingAudio: false,
    speechRecognition: null,
    lastUnderstanding: null,
    lastRecommendations: null,
    catalogueData: [],
    lastSessionId: null,
    lastPrompt: null,
    lastOutput: null,
    lastContextDNA: null,
    lastVerification: null,
    lastTrace: null
  };

  // DOM Elements
  const elements = {
    // Nav & Auth
    tabStudioBtn: document.getElementById('tab-studio-btn'),
    tabDashboardBtn: document.getElementById('tab-dashboard-btn'),
    tabCatalogueBtn: document.getElementById('tab-catalogue-btn'),
    tabHistoryBtn: document.getElementById('tab-history-btn'),
    tabSettingsBtn: document.getElementById('tab-settings-btn'),
    viewStudio: document.getElementById('view-studio'),
    viewDashboard: document.getElementById('view-dashboard'),
    viewCatalogue: document.getElementById('view-catalogue'),
    viewHistory: document.getElementById('view-history'),
    viewSettings: document.getElementById('view-settings'),
    authControls: document.getElementById('auth-controls'),
    authModal: document.getElementById('auth-modal'),
    authModalClose: document.getElementById('auth-modal-close'),
    tabLoginBtn: document.getElementById('tab-login-btn'),
    tabRegisterBtn: document.getElementById('tab-register-btn'),
    loginForm: document.getElementById('login-form'),
    registerForm: document.getElementById('register-form'),
    resetForm: document.getElementById('reset-form'),
    authErrorBanner: document.getElementById('auth-error-banner'),
    forgotPasswordLink: document.getElementById('forgot-password-link'),
    cancelResetBtn: document.getElementById('cancel-reset-btn'),
    launchDemoBtn: document.getElementById('launch-demo-btn'),
    systemStatusText: document.getElementById('system-status-text'),

    // Studio Input
    promptInput: document.getElementById('user-prompt-input'),
    charWordCounter: document.getElementById('char-word-counter'),
    imageFileInput: document.getElementById('image-file-input'),
    imagePreviewContainer: document.getElementById('image-preview-container'),
    imagePreviewThumb: document.getElementById('image-preview-thumb'),
    imagePreviewName: document.getElementById('image-preview-name'),
    imagePreviewSize: document.getElementById('image-preview-size'),
    removeImageBtn: document.getElementById('remove-image-btn'),
    voiceRecordBtn: document.getElementById('voice-record-btn'),
    voiceBtnLabel: document.getElementById('voice-btn-label'),
    audioRecordingBar: document.getElementById('audio-recording-bar'),
    stopAudioBtn: document.getElementById('stop-audio-btn'),
    clearPromptBtn: document.getElementById('clear-prompt-btn'),
    analyzeRouteBtn: document.getElementById('analyze-route-btn'),

    // Understanding
    undConfidenceBadge: document.getElementById('und-confidence-badge'),
    undTaskCategory: document.getElementById('und-task-category'),
    undModalities: document.getElementById('und-modalities'),
    undComplexity: document.getElementById('und-complexity'),
    undTools: document.getElementById('und-tools'),
    undRationaleText: document.getElementById('und-rationale-text'),

    // Recommendations
    recCountBadge: document.getElementById('rec-count-badge'),
    recSummaryBar: document.getElementById('rec-summary-bar'),
    candidatesContainer: document.getElementById('candidates-container'),
    rankTabBtns: document.querySelectorAll('.rank-tab-btn'),

    // Execution
    execStatusPill: document.getElementById('exec-status-pill'),
    execTimePill: document.getElementById('exec-time-pill'),
    execCostPill: document.getElementById('exec-cost-pill'),
    executionLoadingBox: document.getElementById('execution-loading-box'),
    toolAuditBox: document.getElementById('tool-audit-box'),
    toolAuditToggle: document.getElementById('tool-audit-toggle'),
    toolCountNum: document.getElementById('tool-count-num'),
    toolAuditContent: document.getElementById('tool-audit-content'),
    outputContentArea: document.getElementById('output-content-area'),
    outputFooter: document.getElementById('output-footer'),
    engineUsedTag: document.getElementById('engine-used-tag'),
    copyOutputBtn: document.getElementById('copy-output-btn'),

    // Catalogue & History
    catalogueTableBody: document.getElementById('catalogue-table-body'),
    catalogueSearchInput: document.getElementById('catalogue-search-input'),
    catalogueFilterPills: document.querySelectorAll('.filter-pill'),
    historyTableBody: document.getElementById('history-table-body'),

    // Review Feedback Card
    reviewFeedbackCard: document.getElementById('review-feedback-card'),
    reviewTaskCategory: document.getElementById('review-task-category'),
    starRatingRow: document.getElementById('star-rating-row'),
    ratingLabelText: document.getElementById('rating-label-text'),
    diagCorrect: document.getElementById('diag-correct'),
    diagUseful: document.getElementById('diag-useful'),
    diagComplete: document.getElementById('diag-complete'),
    diagFast: document.getElementById('diag-fast'),
    diagCost: document.getElementById('diag-cost'),
    reviewNotesInput: document.getElementById('review-notes-input'),
    submitReviewBtn: document.getElementById('submit-review-btn'),
    reviewFeedbackBanner: document.getElementById('review-feedback-banner'),

    // Token Intelligence Card
    tokenIntelligenceCard: document.getElementById('token-intelligence-card'),
    tokenCardHeaderToggle: document.getElementById('token-card-header-toggle'),
    tokenSavingsPill: document.getElementById('token-savings-pill'),
    tokenStrategyPill: document.getElementById('token-strategy-pill'),
    tokenChevron: document.getElementById('token-chevron'),
    tokenBody: document.getElementById('token-body'),
    tokenBaselineNum: document.getElementById('token-baseline-num'),
    tokenPlannedNum: document.getElementById('token-planned-num'),
    tokenObservedNum: document.getElementById('token-observed-num'),
    tokenSavingsExact: document.getElementById('token-savings-exact'),
    tbarInput: document.getElementById('tbar-input'),
    tbarSystem: document.getElementById('tbar-system'),
    tbarContext: document.getElementById('tbar-context'),
    tbarOutput: document.getElementById('tbar-output'),
    tbarTrust: document.getElementById('tbar-trust'),
    tokenDecompRationale: document.getElementById('token-decomp-rationale'),

    // Dashboard Elements
    kpiTotalRequests: document.getElementById('kpi-total-requests'),
    kpiSuccessRate: document.getElementById('kpi-success-rate'),
    kpiFailedCount: document.getElementById('kpi-failed-count'),
    kpiTotalTokens: document.getElementById('kpi-total-tokens'),
    kpiSavedTokens: document.getElementById('kpi-saved-tokens'),
    kpiTotalCost: document.getElementById('kpi-total-cost'),
    kpiFreeCount: document.getElementById('kpi-free-count'),
    kpiPaidCount: document.getElementById('kpi-paid-count'),
    kpiMedianLatency: document.getElementById('kpi-median-latency'),
    kpiP95Latency: document.getElementById('kpi-p95-latency'),
    kpiAvgLatency: document.getElementById('kpi-avg-latency'),
    kpiAvgRating: document.getElementById('kpi-avg-rating'),
    kpiReviewsCount: document.getElementById('kpi-reviews-count'),
    modelUsageList: document.getElementById('model-usage-list'),
    categoryUsageList: document.getElementById('category-usage-list'),

    // Settings Elements
    saveSettingsBtn: document.getElementById('save-settings-btn'),
    settingsSavedIndicator: document.getElementById('settings-saved-indicator'),
    prefMaxBudget: document.getElementById('pref-max-budget'),
    prefPreferFree: document.getElementById('pref-prefer-free'),
    prefTrustPolicy: document.getElementById('pref-trust-policy'),

    // Post-Response Recovery Controls
    recoveryControlsCard: document.getElementById('recovery-controls-card'),
    btnRetryTenacity: document.getElementById('btn-retry-tenacity'),
    btnImproveOutput: document.getElementById('btn-improve-output'),
    btnVerifyTrust: document.getElementById('btn-verify-trust'),
    btnSwitchEngine: document.getElementById('btn-switch-engine'),
    btnReportIssue: document.getElementById('btn-report-issue'),

    // Trust Layer Elements
    trustLayerCard: document.getElementById('trust-layer-card'),
    trustDecisionBadge: document.getElementById('trust-decision-badge'),
    trustConfidenceBadge: document.getElementById('trust-confidence-badge'),
    pillStageA: document.getElementById('pill-stage-a'),
    pillStageB: document.getElementById('pill-stage-b'),
    pillStageC: document.getElementById('pill-stage-c'),
    pillStageD: document.getElementById('pill-stage-d'),
    pillStageE: document.getElementById('pill-stage-e'),
    trustFindingsCount: document.getElementById('trust-findings-count'),
    trustFindingsList: document.getElementById('trust-findings-list'),

    // Context DNA Elements
    contextDnaCard: document.getElementById('context-dna-card'),
    dnaCompressionPill: document.getElementById('dna-compression-pill'),
    dnaRetentionPill: document.getElementById('dna-retention-pill'),
    dnaHeaderToggle: document.getElementById('context-dna-header-toggle'),
    dnaBody: document.getElementById('context-dna-body'),
    dnaTabs: document.querySelectorAll('.dna-tab'),
    dnaContentDisplay: document.getElementById('dna-content-display'),

    // Execution Trace Timeline
    traceCard: document.getElementById('trace-card'),
    traceTotalLatency: document.getElementById('trace-total-latency'),
    traceTimelineList: document.getElementById('trace-timeline-list'),

    // Jury Demo Elements
    navJuryDemoBtn: document.getElementById('nav-jury-demo-btn'),
    juryDemoModal: document.getElementById('jury-demo-modal'),
    juryModalClose: document.getElementById('jury-modal-close'),
    juryPromptInput: document.getElementById('jury-prompt-input'),
    jurySourceModel: document.getElementById('jury-source-model'),
    juryFailureType: document.getElementById('jury-failure-type'),
    juryFreeOnlyToggle: document.getElementById('jury-free-only-toggle'),
    juryRunFailoverBtn: document.getElementById('jury-run-failover-btn'),
    juryLiveResults: document.getElementById('jury-live-results'),
    juryStatusDot: document.getElementById('jury-status-dot'),
    juryResultsTitle: document.getElementById('jury-results-title'),
    juryResultsLatency: document.getElementById('jury-results-latency'),
    juryDnaPreview: document.getElementById('jury-dna-preview'),
    juryOutputPreview: document.getElementById('jury-output-preview'),
    juryPresetBtns: document.querySelectorAll('.jury-preset-btn')
  };

  // =========================================================================
  // API Fetch Helper
  // =========================================================================
  async function apiRequest(endpoint, method = 'GET', data = null) {
    const headers = { 'Content-Type': 'application/json' };
    if (state.token) {
      headers['Authorization'] = `Bearer ${state.token}`;
    }

    const config = { method, headers };
    if (data && method !== 'GET') {
      config.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(endpoint, config);
      let resData;
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        try {
          resData = await response.json();
        } catch (_) {
          resData = { detail: await response.text() };
        }
      } else {
        const text = await response.text();
        resData = { detail: text };
      }

      if (!response.ok) {
        const errorMsg = (resData && (resData.detail || resData.message)) ||
          (typeof resData === 'string' ? resData : 'API request failed.');
        throw new Error(errorMsg);
      }
      return resData;
    } catch (err) {
      console.error(`API Error on ${endpoint}:`, err);
      throw err;
    }
  }

  // =========================================================================
  // Authentication Management
  // =========================================================================
  async function checkAuthSession() {
    if (!state.token) {
      renderUnauthenticatedControls();
      return;
    }

    try {
      const profile = await apiRequest('/api/v1/auth/me');
      state.user = profile;
      if (profile.preferences && profile.preferences.ranking_mode) {
        state.currentMode = profile.preferences.ranking_mode;
        updateActiveRankTab(state.currentMode);
      }
      renderAuthenticatedControls();
    } catch (err) {
      // Session expired
      logout();
    }
  }

  function renderUnauthenticatedControls() {
    elements.authControls.innerHTML = `
      <button class="auth-btn-primary" id="open-auth-modal-btn">Sign In / Demo</button>
    `;
    document.getElementById('open-auth-modal-btn').onclick = () => openAuthModal('login');
  }

  function renderAuthenticatedControls() {
    const isDemo = state.user && state.user.is_demo;
    elements.authControls.innerHTML = `
      <div class="user-profile-menu">
        <span class="user-avatar-tag" title="${state.user.email}">
          ${isDemo ? '✨ Demo Pilot' : state.user.name}
        </span>
        <button class="logout-icon-btn" id="logout-btn" title="Sign Out">Logout</button>
      </div>
    `;
    document.getElementById('logout-btn').onclick = logout;
  }

  function logout() {
    state.token = null;
    state.user = null;
    localStorage.removeItem('syntra_token');
    renderUnauthenticatedControls();
  }

  function openAuthModal(mode = 'login') {
    elements.authModal.classList.remove('hidden');
    elements.authErrorBanner.classList.add('hidden');
    switchAuthForm(mode);
  }

  function closeAuthModal() {
    elements.authModal.classList.add('hidden');
  }

  function switchAuthForm(mode) {
    elements.authErrorBanner.classList.add('hidden');
    elements.tabLoginBtn.classList.toggle('active', mode === 'login');
    elements.tabRegisterBtn.classList.toggle('active', mode === 'register');

    elements.loginForm.classList.toggle('hidden', mode !== 'login');
    elements.registerForm.classList.toggle('hidden', mode !== 'register');
    elements.resetForm.classList.toggle('hidden', mode !== 'reset');

    if (mode === 'login') {
      document.getElementById('auth-modal-title').textContent = 'Sign In to OpenPilot';
      document.getElementById('auth-modal-subtitle').textContent = 'One request. Every model. Zero guesswork.';
    } else if (mode === 'register') {
      document.getElementById('auth-modal-title').textContent = 'Create OpenPilot Account';
      document.getElementById('auth-modal-subtitle').textContent = 'Join the next-generation AI orchestration harness.';
    } else if (mode === 'reset') {
      document.getElementById('auth-modal-title').textContent = 'Reset Password';
      document.getElementById('auth-modal-subtitle').textContent = 'Enter your email and new password.';
    }
  }

  function showAuthError(msg) {
    elements.authErrorBanner.textContent = msg;
    elements.authErrorBanner.classList.remove('hidden');
  }

  // =========================================================================
  // Health & System Probe
  // =========================================================================
  async function probeSystemHealth() {
    try {
      const health = await apiRequest('/health');
      elements.systemStatusText.textContent = `${health.available_models} Engines Online`;
    } catch (err) {
      elements.systemStatusText.textContent = 'Offline Fallback';
    }
  }

  // =========================================================================
  // Studio Navigation & View Switching
  // =========================================================================
  function switchView(tabName) {
    state.activeTab = tabName;
    if (elements.tabStudioBtn) elements.tabStudioBtn.classList.toggle('active', tabName === 'studio');
    if (elements.tabDashboardBtn) elements.tabDashboardBtn.classList.toggle('active', tabName === 'dashboard');
    if (elements.tabCatalogueBtn) elements.tabCatalogueBtn.classList.toggle('active', tabName === 'catalogue');
    if (elements.tabHistoryBtn) elements.tabHistoryBtn.classList.toggle('active', tabName === 'history');
    if (elements.tabSettingsBtn) elements.tabSettingsBtn.classList.toggle('active', tabName === 'settings');

    if (elements.viewStudio) elements.viewStudio.classList.toggle('active', tabName === 'studio');
    if (elements.viewDashboard) elements.viewDashboard.classList.toggle('active', tabName === 'dashboard');
    if (elements.viewCatalogue) elements.viewCatalogue.classList.toggle('active', tabName === 'catalogue');
    if (elements.viewHistory) elements.viewHistory.classList.toggle('active', tabName === 'history');
    if (elements.viewSettings) elements.viewSettings.classList.toggle('active', tabName === 'settings');

    if (tabName === 'dashboard') {
      loadDashboardStats();
    } else if (tabName === 'catalogue') {
      loadCatalogue();
    } else if (tabName === 'history') {
      loadHistory();
    } else if (tabName === 'settings') {
      loadSettings();
    }
  }

  // =========================================================================
  // Multimodal Inputs & Word Counter
  // =========================================================================
  function updateInputCounter() {
    const text = elements.promptInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const chars = text.length;
    elements.charWordCounter.textContent = `${words} words · ${chars} chars`;
  }

  function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG, or WEBP).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('Image file size must be less than 10MB.');
      return;
    }

    const reader = new FileReader();
    reader.onload = function (event) {
      state.imageAttachment = {
        base64: event.target.result,
        name: file.name,
        size: `${Math.round(file.size / 1024)} KB`
      };
      elements.imagePreviewThumb.src = event.target.result;
      elements.imagePreviewName.textContent = file.name;
      elements.imagePreviewSize.textContent = state.imageAttachment.size;
      elements.imagePreviewContainer.classList.remove('hidden');

      // Auto update request understanding with image flag
      triggerUnderstanding();
    };
    reader.readAsDataURL(file);
  }

  function removeImageAttachment() {
    state.imageAttachment = null;
    elements.imageFileInput.value = '';
    elements.imagePreviewContainer.classList.add('hidden');
    triggerUnderstanding();
  }

  // Voice Recording via Web Speech API
  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech Recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.');
      return null;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = function () {
      state.isRecordingAudio = true;
      elements.voiceRecordBtn.classList.add('recording');
      elements.voiceBtnLabel.textContent = 'Listening...';
      elements.audioRecordingBar.classList.remove('hidden');
    };

    recognition.onresult = function (event) {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      if (finalTranscript) {
        elements.promptInput.value = (elements.promptInput.value + ' ' + finalTranscript).trim();
        updateInputCounter();
      }
    };

    recognition.onerror = function (event) {
      console.warn('Speech recognition error:', event.error);
      stopAudioRecording();
    };

    recognition.onend = function () {
      stopAudioRecording();
    };

    return recognition;
  }

  function toggleAudioRecording() {
    if (state.isRecordingAudio) {
      stopAudioRecording();
    } else {
      if (!state.speechRecognition) {
        state.speechRecognition = initSpeechRecognition();
      }
      if (state.speechRecognition) {
        try {
          state.speechRecognition.start();
        } catch (e) {
          console.error(e);
        }
      }
    }
  }

  function stopAudioRecording() {
    if (state.speechRecognition && state.isRecordingAudio) {
      try {
        state.speechRecognition.stop();
      } catch (e) {}
    }
    state.isRecordingAudio = false;
    elements.voiceRecordBtn.classList.remove('recording');
    elements.voiceBtnLabel.textContent = 'Voice';
    elements.audioRecordingBar.classList.add('hidden');
    triggerUnderstanding();
  }

  // =========================================================================
  // Dynamic Request Understanding & Recommendations
  // =========================================================================
  async function triggerUnderstanding() {
    const prompt = elements.promptInput.value.trim();
    if (!prompt) return;

    try {
      const understanding = await apiRequest('/api/v1/understand', 'POST', {
        prompt,
        has_image: Boolean(state.imageAttachment),
        has_audio: state.isRecordingAudio
      });
      state.lastUnderstanding = understanding;
      renderUnderstanding(understanding);

      // Re-calculate recommendations
      triggerRecommendations();
    } catch (err) {
      console.error('Understanding error:', err);
    }
  }

  function renderUnderstanding(u) {
    elements.undConfidenceBadge.textContent = `${Math.round(u.confidence * 100)}% Confidence`;
    elements.undTaskCategory.textContent = u.category_label;
    elements.undComplexity.textContent = u.complexity.toUpperCase();
    elements.undTools.textContent = u.required_tools.length ? u.required_tools.join(', ') : 'None';
    elements.undRationaleText.textContent = u.rationale;

    elements.undModalities.innerHTML = u.required_modalities
      .map(m => `<span class="tag">${m}</span>`)
      .join('');
  }

  async function triggerRecommendations() {
    const prompt = elements.promptInput.value.trim() || 'General query';
    try {
      const recResult = await apiRequest('/api/v1/recommend', 'POST', {
        prompt,
        ranking_mode: state.currentMode,
        has_image: Boolean(state.imageAttachment),
        has_audio: state.isRecordingAudio
      });
      state.lastRecommendations = recResult;
      renderRecommendations(recResult);
    } catch (err) {
      console.error('Recommendation error:', err);
    }
  }

  function renderRecommendations(res) {
    elements.recCountBadge.textContent = `${res.total_eligible_models} Engines Evaluated`;
    elements.recSummaryBar.textContent = res.summary_rationale;

    elements.candidatesContainer.innerHTML = res.models.map(m => `
      <div class="candidate-card ${m.is_top_pick ? 'top-pick' : ''}" data-model-id="${m.model_id}">
        <div class="candidate-header">
          <div class="cand-title-group">
            <span class="rank-pill">#${m.rank}</span>
            <span class="cand-name">${m.model_name}</span>
            <span style="font-size:0.75rem; color:var(--text-muted)">(${m.provider})</span>
          </div>
          ${m.is_top_pick ? '<span class="top-badge">TOP PICK</span>' : ''}
        </div>

        <div class="cand-meta-row">
          <span class="pricing-tag">${m.pricing_display}</span>
          <span>·</span>
          <span class="score-tag">${m.benchmark_name}: ${m.benchmark_score}%</span>
          <span>·</span>
          <span>Context: ${Math.round(m.context_limit_tokens / 1024)}k</span>
        </div>

        <p class="cand-reason">${m.suitability_reason}</p>

        <div class="cand-footer">
          <span class="readiness-pill ${m.execution_status}">${m.execution_status_label}</span>
          <button class="run-model-btn" data-run-model="${m.model_id}">Run Engine</button>
        </div>
      </div>
    `).join('');

    // Attach click listeners to Run buttons
    document.querySelectorAll('[data-run-model]').forEach(btn => {
      btn.onclick = (e) => {
        const modelId = e.currentTarget.getAttribute('data-run-model');
        executeRequest(modelId);
      };
    });
  }

  function updateActiveRankTab(mode) {
    state.currentMode = mode;
    elements.rankTabBtns.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
    });
    triggerRecommendations();
  }

  // =========================================================================
  // Real Model Execution
  // =========================================================================
  async function executeRequest(selectedModel = null) {
    const prompt = elements.promptInput.value.trim();
    if (!prompt) {
      alert('Please enter a request prompt before executing.');
      return;
    }

    // Prepare UI for execution
    elements.executionLoadingBox.classList.remove('hidden');
    elements.toolAuditBox.classList.add('hidden');
    elements.outputFooter.classList.add('hidden');
    elements.outputContentArea.innerHTML = '';
    elements.execStatusPill.classList.add('hidden');
    elements.execTimePill.classList.add('hidden');
    elements.execCostPill.classList.add('hidden');

    try {
      const payload = {
        prompt,
        selected_model: selectedModel,
        ranking_mode: state.currentMode,
        image_base64: state.imageAttachment ? state.imageAttachment.base64 : null,
        has_audio: state.isRecordingAudio
      };

      const result = await apiRequest('/api/v1/execute', 'POST', payload);

      // Render Metrics
      elements.execStatusPill.textContent = result.status;
      elements.execStatusPill.classList.remove('hidden');
      elements.execTimePill.textContent = `${result.execution_time_ms} ms`;
      elements.execTimePill.classList.remove('hidden');
      elements.execCostPill.textContent = `$${result.cost_usd.toFixed(6)}`;
      elements.execCostPill.classList.remove('hidden');

      // Render Tools if any
      if (result.executed_tools && result.executed_tools.length > 0) {
        elements.toolCountNum.textContent = result.executed_tools.length;
        elements.toolAuditContent.innerHTML = result.executed_tools.map(t => `
          <div class="tool-run-item">
            <strong>${t.tool_name}</strong>: ${JSON.stringify(t.output)}
            <span style="color:var(--text-muted); margin-left:8px">(${t.execution_time_ms} ms)</span>
          </div>
        `).join('');
        elements.toolAuditBox.classList.remove('hidden');
      }

      // Render Output Text
      elements.outputContentArea.innerHTML = formatMarkdown(result.final_output);
      elements.engineUsedTag.textContent = `Engine: ${result.selected_model}`;
      elements.outputFooter.classList.remove('hidden');

      // Save State
      state.lastSessionId = result.session_id;
      state.lastPrompt = prompt;
      state.lastOutput = result.final_output;
      state.lastVerification = result.verification;
      state.lastContextDNA = result.context_dna;
      state.lastTrace = result.trace;

      // Render Trust Layer Verification (Part 4)
      if (result.verification) {
        renderTrustLayer(result.verification);
      }

      // Render Context DNA (Part 7 & 8)
      if (result.context_dna) {
        renderContextDNA(result.context_dna);
      }

      // Render Execution Trace Timeline (Part 12)
      if (result.trace) {
        renderExecutionTrace(result.trace);
      }

      // Render Token Intelligence & Adaptive Budget (Section 8)
      if (result.token_plan || result.token_accounting) {
        renderTokenIntelligence(result.token_plan, result.token_accounting);
      }

      // Render Review & Routing Feedback Loop Card (Section 12)
      renderReviewFeedbackCard(result);

      // Show Post-Response Quality Recovery Controls (Part 10)
      if (elements.recoveryControlsCard) {
        elements.recoveryControlsCard.classList.remove('hidden');
      }

    } catch (err) {
      elements.outputContentArea.innerHTML = `
        <div style="color:var(--accent-rose); padding:1rem; background:rgba(244,63,94,0.1); border-radius:8px">
          <strong>Execution Error:</strong> ${err.message}
        </div>
      `;
    } finally {
      elements.executionLoadingBox.classList.add('hidden');
    }
  }

  // =========================================================================
  // Trust Layer Output Verification Rendering
  // =========================================================================
  function renderTrustLayer(v) {
    if (!elements.trustLayerCard || !v) return;
    elements.trustLayerCard.classList.remove('hidden');

    elements.trustDecisionBadge.textContent = v.decision;
    elements.trustDecisionBadge.className = 'trust-decision-pill ' + v.decision.toLowerCase();
    elements.trustConfidenceBadge.textContent = `${Math.round(v.confidence_score * 100)}% Conf (${v.verification_latency_ms}ms)`;

    setStageStatus(elements.pillStageA, v.stage_a_integrity);
    setStageStatus(elements.pillStageB, v.stage_b_compliance);
    setStageStatus(elements.pillStageC, v.stage_c_task_validation);
    setStageStatus(elements.pillStageD, v.stage_d_independent_review);
    setStageStatus(elements.pillStageE, v.passed);

    const passedCount = (v.findings || []).filter(f => f.status === 'PASS').length;
    elements.trustFindingsCount.textContent = `${passedCount}/${(v.findings || []).length} Checks Passed`;

    if (v.findings && v.findings.length > 0) {
      elements.trustFindingsList.innerHTML = v.findings.map(f => `
        <div class="finding-item">
          <span class="finding-tag ${f.severity.toLowerCase()}">${f.status}</span>
          <span><strong>${escapeHtml(f.check_name)}:</strong> ${escapeHtml(f.description)} ${f.evidence ? `<em>(${escapeHtml(f.evidence)})</em>` : ''}</span>
        </div>
      `).join('');
    } else {
      elements.trustFindingsList.innerHTML = '<div class="finding-item"><span class="finding-tag info">INFO</span><span>All verification checks passed with no detected errors.</span></div>';
    }
  }

  function setStageStatus(el, passed) {
    if (!el) return;
    el.classList.toggle('passed', !!passed);
    el.classList.toggle('failed', !passed);
  }

  // =========================================================================
  // Context DNA Inspector Rendering
  // =========================================================================
  let activeDnaTab = 'goal';

  function renderContextDNA(dna) {
    if (!elements.contextDnaCard || !dna) return;
    elements.contextDnaCard.classList.remove('hidden');
    displayDnaTabContent(dna, activeDnaTab);
  }

  function displayDnaTabContent(dna, tab) {
    if (!elements.dnaContentDisplay || !dna) return;

    if (tab === 'goal') {
      elements.dnaContentDisplay.innerHTML = `
        <div class="dna-kv-row"><span class="dna-kv-label">User Objective:</span> ${escapeHtml(dna.objective || '—')}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Intended Outcome:</span> ${escapeHtml(dna.intended_outcome || '—')}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Acceptance Criteria:</span> ${(dna.acceptance_criteria || []).map(c => `• ${escapeHtml(c)}`).join('<br>') || 'None'}</div>
      `;
    } else if (tab === 'constraints') {
      elements.dnaContentDisplay.innerHTML = `
        <div class="dna-kv-row"><span class="dna-kv-label">Format Requirement:</span> ${escapeHtml(dna.format_requirements || 'Standard')}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Mandatory Constraints:</span> ${(dna.explicit_requirements || []).map(r => `• ${escapeHtml(r)}`).join('<br>') || 'None'}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Technical Constraints:</span> ${(dna.technical_constraints || []).join(', ') || 'None'}</div>
      `;
    } else if (tab === 'state') {
      elements.dnaContentDisplay.innerHTML = `
        <div class="dna-kv-row"><span class="dna-kv-label">Current Task:</span> ${escapeHtml(dna.current_task || '—')}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Completed Milestones:</span> ${(dna.completed_milestones || []).map(m => `✓ ${escapeHtml(m)}`).join('<br>') || 'None'}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Immediate Next Action:</span> ${escapeHtml(dna.immediate_next_action || '—')}</div>
      `;
    } else if (tab === 'knowledge') {
      elements.dnaContentDisplay.innerHTML = `
        <div class="dna-kv-row"><span class="dna-kv-label">Confirmed Findings:</span> ${(dna.confirmed_findings || []).map(f => `• ${escapeHtml(f)}`).join('<br>') || 'None'}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Reliable Tool Groundings:</span> ${dna.reliable_tool_outputs ? JSON.stringify(dna.reliable_tool_outputs) : 'None'}</div>
      `;
    } else if (tab === 'defects') {
      elements.dnaContentDisplay.innerHTML = `
        <div class="dna-kv-row"><span class="dna-kv-label">Failed Validations:</span> ${(dna.failed_validations || []).map(v => `⚠️ ${escapeHtml(v)}`).join('<br>') || 'None recorded'}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Known Issues to Avoid:</span> ${(dna.known_issues || []).map(i => `• ${escapeHtml(i)}`).join('<br>') || 'None'}</div>
      `;
    } else if (tab === 'decisions') {
      elements.dnaContentDisplay.innerHTML = `
        <div class="dna-kv-row"><span class="dna-kv-label">Confirmed Architectural Decisions:</span> ${(dna.confirmed_decisions || []).map(d => `✓ ${escapeHtml(d)}`).join('<br>') || 'None'}</div>
        <div class="dna-kv-row"><span class="dna-kv-label">Rejected / Abandoned Approaches:</span> ${(dna.rejected_approaches || []).map(a => `✗ ${escapeHtml(a)}`).join('<br>') || 'None'}</div>
      `;
    } else if (tab === 'artifacts') {
      const snippets = Object.entries(dna.code_artifacts || {});
      if (snippets.length === 0) {
        elements.dnaContentDisplay.innerHTML = '<p>No code or file artifacts generated in this turn.</p>';
      } else {
        elements.dnaContentDisplay.innerHTML = snippets.map(([name, code]) => `
          <div style="margin-bottom:8px">
            <strong>${escapeHtml(name)}:</strong>
            <pre style="background:rgba(0,0,0,0.5); padding:6px; border-radius:4px; font-size:0.7rem; overflow-x:auto"><code>${escapeHtml(code)}</code></pre>
          </div>
        `).join('');
      }
    } else if (tab === 'continuation') {
      elements.dnaContentDisplay.innerHTML = `
        <div class="dna-kv-row"><span class="dna-kv-label">Continuation Mandate for Next Model:</span></div>
        <pre style="background:rgba(0,0,0,0.5); padding:6px; border-radius:4px; font-size:0.7rem; white-space:pre-wrap">${escapeHtml(dna.continuation_prompt || '—')}</pre>
        <div class="dna-kv-row" style="margin-top:6px"><span class="dna-kv-label">Prohibited Repetitions:</span> ${(dna.prohibited_repetitions || []).join('; ') || 'None'}</div>
      `;
    }
  }

  // =========================================================================
  // Execution Trace Timeline Rendering
  // =========================================================================
  function renderExecutionTrace(trace) {
    if (!elements.traceCard || !trace) return;
    elements.traceCard.classList.remove('hidden');
    elements.traceTotalLatency.textContent = `Total Latency: ${trace.total_elapsed_ms || 0} ms`;

    if (trace.events && trace.events.length > 0) {
      elements.traceTimelineList.innerHTML = trace.events.map(ev => `
        <div class="trace-event-row">
          <span class="trace-time-badge">+${ev.elapsed_ms}ms</span>
          <span class="trace-component-badge">${escapeHtml(ev.component)}</span>
          <span class="trace-event-text"><strong>${escapeHtml(ev.event_name)}:</strong> ${escapeHtml(ev.description)}</span>
        </div>
      `).join('');
    }
  }

  function formatMarkdown(text) {
    if (!text) return '<p>No output received.</p>';
    // Convert code blocks
    let html = text.replace(/```([\s\S]*?)```/g, (match, p1) => {
      return `<pre><code>${escapeHtml(p1.trim())}</code></pre>`;
    });
    // Convert bullet points
    html = html.replace(/^\s*[\*\-]\s+(.*)$/gim, '<li>$1</li>');
    // Convert bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Wrap loose lines in paragraphs
    const paragraphs = html.split('\n\n').map(p => {
      if (p.startsWith('<pre>') || p.startsWith('<li>')) return p;
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    });
    return paragraphs.join('');
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // =========================================================================
  // Model Catalogue & History Views
  // =========================================================================
  async function loadCatalogue() {
    try {
      const data = await apiRequest('/api/v1/catalogue');
      const models = Array.isArray(data) ? data : (data.models || []);
      state.catalogueData = models;
      renderCatalogueTable(models);
    } catch (err) {
      console.error(err);
    }
  }

  function renderCatalogueTable(models) {
    elements.catalogueTableBody.innerHTML = models.map(m => `
      <tr>
        <td>
          <strong>${m.name}</strong><br>
          <span style="font-size:0.75rem; color:var(--text-muted)">${m.provider}</span>
        </td>
        <td>
          <span class="tag" style="background:rgba(255,255,255,0.06)">
            ${m.is_free ? 'Free / Open Weights' : 'Commercial API'}
          </span>
        </td>
        <td style="color:var(--accent-emerald); font-weight:600">
          ${m.is_free ? '100% Free' : `$${m.input_cost_per_1m.toFixed(2)} in / $${m.output_cost_per_1m.toFixed(2)} out`}
        </td>
        <td style="font-family:var(--font-mono)">${Math.round(m.context_limit_tokens / 1024)}k tokens</td>
        <td>
          <strong style="color:var(--accent-cyan)">${m.benchmark_name}: ${m.benchmark_score}%</strong><br>
          <span style="font-size:0.7rem; color:var(--text-muted)">${m.benchmark_source}</span><br>
          <span style="font-size:0.68rem; color:#FBBF24; font-weight:600">${m.rating_display || '4.0★ prior'}</span>
        </td>
        <td>
          <span class="readiness-pill ${m.status}">${m.status_label}</span>
        </td>
      </tr>
    `).join('');
  }

  async function loadHistory() {
    try {
      const history = await apiRequest('/api/v1/history');
      if (history.length === 0) {
        elements.historyTableBody.innerHTML = `
          <tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:2rem">No execution turns recorded yet. Run a request in the Studio!</td></tr>
        `;
        return;
      }
      elements.historyTableBody.innerHTML = history.map(h => `
        <tr>
          <td style="font-size:0.75rem; color:var(--text-muted)">${new Date(h.created_at).toLocaleTimeString()}</td>
          <td><span class="tag">${h.category}</span></td>
          <td style="max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">${escapeHtml(h.prompt)}</td>
          <td><strong>${h.selected_model}</strong></td>
          <td style="font-family:var(--font-mono)">${h.latency_ms} ms</td>
          <td style="font-family:var(--font-mono)">${h.tokens_used}</td>
          <td style="color:var(--accent-emerald)">$${h.cost_usd.toFixed(6)}</td>
          <td><span class="readiness-pill ${h.status === 'COMPLETE' ? 'online' : 'config_required'}">${h.status}</span></td>
        </tr>
      `).join('');
    } catch (err) {
      console.error(err);
    }
  }

  // =========================================================================
  // Token Intelligence & Budget Allocation (Section 8)
  // =========================================================================
  function renderTokenIntelligence(plan, accounting) {
    if (!elements.tokenIntelligenceCard || !plan) return;
    elements.tokenIntelligenceCard.classList.remove('hidden');

    if (elements.tokenBaselineNum) elements.tokenBaselineNum.textContent = `${(plan.baseline_tokens_estimate || 0).toLocaleString()} tokens`;
    if (elements.tokenPlannedNum) elements.tokenPlannedNum.textContent = `${(plan.total_planned_tokens || 0).toLocaleString()} tokens`;
    
    if (accounting) {
      if (elements.tokenObservedNum) elements.tokenObservedNum.textContent = `${(accounting.total_tokens || 0).toLocaleString()} tokens`;
      if (elements.tokenSavingsExact) elements.tokenSavingsExact.textContent = `${(accounting.tokens_saved || 0).toLocaleString()} tokens saved (${accounting.savings_percentage || 0}%)`;
      if (elements.tokenSavingsPill) elements.tokenSavingsPill.textContent = `${accounting.savings_percentage}% Saved vs Baseline`;
    } else {
      if (elements.tokenObservedNum) elements.tokenObservedNum.textContent = '—';
      if (elements.tokenSavingsPill) elements.tokenSavingsPill.textContent = `${plan.projected_savings_pct}% Projected Savings`;
    }

    if (elements.tokenStrategyPill) {
      elements.tokenStrategyPill.textContent = (plan.decomposition_strategy || 'single_model').replace(/_/g, ' ').toUpperCase();
    }

    // Lifecycle allocation segment widths
    const total = plan.total_planned_tokens || 1000;
    const pIn = Math.max(5, Math.round(((plan.allocated_input_tokens || 200) / total) * 100));
    const pSys = Math.max(5, Math.round(((plan.allocated_system_tokens || 85) / total) * 100));
    const pCtx = Math.max(5, Math.round(((plan.allocated_context_tokens || 150) / total) * 100));
    const pOut = Math.max(10, Math.round(((plan.allocated_output_tokens || 400) / total) * 100));
    const pTrust = Math.max(5, 100 - (pIn + pSys + pCtx + pOut));

    if (elements.tbarInput) { elements.tbarInput.style.width = `${pIn}%`; elements.tbarInput.textContent = `Input (${pIn}%)`; }
    if (elements.tbarSystem) { elements.tbarSystem.style.width = `${pSys}%`; elements.tbarSystem.textContent = `Sys (${pSys}%)`; }
    if (elements.tbarContext) { elements.tbarContext.style.width = `${pCtx}%`; elements.tbarContext.textContent = `Ctx (${pCtx}%)`; }
    if (elements.tbarOutput) { elements.tbarOutput.style.width = `${pOut}%`; elements.tbarOutput.textContent = `Out (${pOut}%)`; }
    if (elements.tbarTrust) { elements.tbarTrust.style.width = `${pTrust}%`; elements.tbarTrust.textContent = `Res (${pTrust}%)`; }

    if (elements.tokenDecompRationale) {
      elements.tokenDecompRationale.textContent = `Decomposition Rationale: ${plan.decomposition_rationale || 'Single engine inference satisfies accuracy and latency bounds.'}`;
    }
  }

  // =========================================================================
  // Review & Recommendation Feedback Loop (Section 12)
  // =========================================================================
  let currentRating = 5;

  function renderReviewFeedbackCard(result) {
    if (!elements.reviewFeedbackCard) return;
    elements.reviewFeedbackCard.classList.remove('hidden');

    const catLabel = result.understanding ? result.understanding.category_label : 'this task';
    if (elements.reviewTaskCategory) elements.reviewTaskCategory.textContent = catLabel;
    if (elements.reviewFeedbackBanner) elements.reviewFeedbackBanner.classList.add('hidden');

    // Default 5 stars
    setStarRating(5);

    // Bind star clicks
    if (elements.starRatingRow) {
      elements.starRatingRow.querySelectorAll('.star-btn').forEach(btn => {
        btn.onclick = () => {
          const val = parseInt(btn.getAttribute('data-star'), 10);
          setStarRating(val);
        };
      });
    }

    // Submit action
    if (elements.submitReviewBtn) {
      elements.submitReviewBtn.onclick = async () => {
        try {
          elements.submitReviewBtn.disabled = true;
          elements.submitReviewBtn.textContent = 'Submitting...';

          const res = await apiRequest('/api/v1/reviews', 'POST', {
            session_id: state.lastSessionId || 'sess_default',
            model_id: result.selected_model,
            provider: result.model_spec ? result.model_spec.provider : 'Configured',
            task_category: result.understanding ? result.understanding.task_category : 'general',
            rating: currentRating,
            is_correct: elements.diagCorrect ? elements.diagCorrect.checked : true,
            is_useful: elements.diagUseful ? elements.diagUseful.checked : true,
            is_complete: elements.diagComplete ? elements.diagComplete.checked : true,
            is_fast: elements.diagFast ? elements.diagFast.checked : true,
            cost_acceptable: elements.diagCost ? elements.diagCost.checked : true,
            verification_status: result.verification ? result.verification.decision : 'VERIFIED',
            feedback_text: elements.reviewNotesInput ? elements.reviewNotesInput.value.trim() : '',
            latency_ms: result.execution_time_ms || 0,
            cost_usd: result.cost_usd || 0
          });

          const stats = res.updated_model_stats;
          if (elements.reviewFeedbackBanner) {
            elements.reviewFeedbackBanner.innerHTML = `
              <strong>✓ Rating Persisted & Integrated!</strong><br>
              Bayesian smoothed rating for <strong>${result.selected_model}</strong> updated to 
              <strong>${stats.smoothed_rating}★</strong> across <strong>${stats.review_count}</strong> recorded reviews.<br>
              Future routing recommendations for <em>${catLabel}</em> now dynamically incorporate this feedback!
            `;
            elements.reviewFeedbackBanner.classList.remove('hidden');
          }
          elements.submitReviewBtn.textContent = 'Updated ✓';
        } catch (err) {
          alert(`Failed to submit review: ${err.message}`);
          elements.submitReviewBtn.textContent = 'Submit Rating';
        } finally {
          elements.submitReviewBtn.disabled = false;
        }
      };
    }
  }

  function setStarRating(val) {
    currentRating = val;
    if (elements.starRatingRow) {
      elements.starRatingRow.querySelectorAll('.star-btn').forEach(btn => {
        const star = parseInt(btn.getAttribute('data-star'), 10);
        btn.classList.toggle('active', star <= val);
      });
    }
    const labels = {
      1: '1★ — Unsatisfactory / Inaccurate',
      2: '2★ — Subpar / Format Issues',
      3: '3★ — Acceptable / Partial',
      4: '4★ — Good / Meets Requirements',
      5: '5★ — Exceptional / Verified High Quality'
    };
    if (elements.ratingLabelText) elements.ratingLabelText.textContent = labels[val] || `${val} Stars`;
  }

  // =========================================================================
  // Control Center / Live Dashboard (Section 15)
  // =========================================================================
  async function loadDashboardStats() {
    try {
      const stats = await apiRequest('/api/v1/dashboard/stats');
      if (elements.kpiTotalRequests) elements.kpiTotalRequests.textContent = stats.total_requests;
      if (elements.kpiSuccessRate) elements.kpiSuccessRate.textContent = `${stats.success_rate_pct}%`;
      if (elements.kpiFailedCount) elements.kpiFailedCount.textContent = stats.failed_executions;
      if (elements.kpiTotalTokens) elements.kpiTotalTokens.textContent = (stats.total_tokens_consumed || 0).toLocaleString();
      if (elements.kpiSavedTokens) elements.kpiSavedTokens.textContent = `+${(stats.estimated_tokens_saved || 0).toLocaleString()}`;
      if (elements.kpiTotalCost) elements.kpiTotalCost.textContent = `$${(stats.total_cost_usd || 0).toFixed(6)}`;
      if (elements.kpiFreeCount) elements.kpiFreeCount.textContent = stats.free_vs_paid ? stats.free_vs_paid.free : 0;
      if (elements.kpiPaidCount) elements.kpiPaidCount.textContent = stats.free_vs_paid ? stats.free_vs_paid.paid : 0;

      if (stats.latency) {
        if (elements.kpiMedianLatency) elements.kpiMedianLatency.textContent = `${stats.latency.median_ms} ms`;
        if (elements.kpiP95Latency) elements.kpiP95Latency.textContent = `${stats.latency.p95_ms} ms`;
        if (elements.kpiAvgLatency) elements.kpiAvgLatency.textContent = `${stats.latency.avg_ms} ms`;
      }

      if (elements.kpiAvgRating) elements.kpiAvgRating.textContent = `${stats.average_rating || 0}★`;
      if (elements.kpiReviewsCount) elements.kpiReviewsCount.textContent = stats.reviews_count || 0;

      // Model Usage Distribution
      if (elements.modelUsageList && stats.model_usage_distribution) {
        const entries = Object.entries(stats.model_usage_distribution);
        const maxVal = Math.max(...entries.map(e => e[1]), 1);
        if (entries.length === 0) {
          elements.modelUsageList.innerHTML = '<p style="color:var(--text-muted); font-size:0.8rem">No executions recorded yet.</p>';
        } else {
          elements.modelUsageList.innerHTML = entries.map(([model, count]) => {
            const pct = Math.round((count / maxVal) * 100);
            return `
              <div class="dist-item">
                <div class="dist-header">
                  <span>${escapeHtml(model)}</span>
                  <span>${count} turn${count !== 1 ? 's' : ''}</span>
                </div>
                <div class="dist-bar-wrap">
                  <div class="dist-fill" style="width: ${pct}%;"></div>
                </div>
              </div>
            `;
          }).join('');
        }
      }

      // Task Category Distribution
      if (elements.categoryUsageList && stats.category_distribution) {
        const entries = Object.entries(stats.category_distribution);
        const maxVal = Math.max(...entries.map(e => e[1]), 1);
        if (entries.length === 0) {
          elements.categoryUsageList.innerHTML = '<p style="color:var(--text-muted); font-size:0.8rem">No category data recorded yet.</p>';
        } else {
          elements.categoryUsageList.innerHTML = entries.map(([cat, count]) => {
            const pct = Math.round((count / maxVal) * 100);
            return `
              <div class="dist-item">
                <div class="dist-header">
                  <span>${escapeHtml(cat.replace(/_/g, ' '))}</span>
                  <span>${count}</span>
                </div>
                <div class="dist-bar-wrap">
                  <div class="dist-fill" style="width: ${pct}%; background: linear-gradient(90deg, #38BDF8, #10B981);"></div>
                </div>
              </div>
            `;
          }).join('');
        }
      }

    } catch (err) {
      console.error('Failed to load dashboard statistics:', err);
    }
  }

  // =========================================================================
  // Settings & Preferences (Section 17.G)
  // =========================================================================
  async function loadSettings() {
    const raw = localStorage.getItem('syntra_user_preferences');
    if (raw) {
      try {
        const prefs = JSON.parse(raw);
        if (prefs.ranking_mode) {
          const radio = document.querySelector(`input[name="pref-ranking-mode"][value="${prefs.ranking_mode}"]`);
          if (radio) radio.checked = true;
          state.currentMode = prefs.ranking_mode;
        }
        const maxBudgetInput = document.getElementById('pref-max-budget');
        if (prefs.max_budget && maxBudgetInput) maxBudgetInput.value = prefs.max_budget;
        const preferFreeInput = document.getElementById('pref-prefer-free');
        if (prefs.prefer_free !== undefined && preferFreeInput) preferFreeInput.checked = prefs.prefer_free;
        const trustPolicySelect = document.getElementById('pref-trust-policy');
        if (prefs.trust_policy && trustPolicySelect) trustPolicySelect.value = prefs.trust_policy;
      } catch (e) {
        console.error(e);
      }
    }

    // Fetch live API key configuration and installed local models
    try {
      const keysData = await apiRequest('/api/v1/settings/keys');
      const geminiStatus = document.getElementById('gemini-status-badge');
      const geminiInput = document.getElementById('pref-gemini-key');
      if (geminiStatus) {
        if (keysData.gemini.configured) {
          geminiStatus.textContent = `✓ Live & Connected (${keysData.gemini.masked})`;
          geminiStatus.style.color = '#10b981';
          if (geminiInput) geminiInput.placeholder = 'Key configured (enter new key to replace)';
        } else {
          geminiStatus.textContent = 'Key Required (Paste your Gemini API key below)';
          geminiStatus.style.color = '#f59e0b';
        }
      }

      const ollamaBadge = document.getElementById('ollama-status-badge');
      if (ollamaBadge) {
        ollamaBadge.textContent = keysData.ollama.online ? 'Active & Ready (localhost:11434)' : 'Offline Fallback';
        ollamaBadge.style.color = keysData.ollama.online ? '#10b981' : '#f43f5e';
      }

      const localList = document.getElementById('installed-local-models-list');
      if (localList && keysData.ollama.installed_models && keysData.ollama.installed_models.length > 0) {
        localList.innerHTML = keysData.ollama.installed_models.map(m => `
          <span class="tag" style="background: rgba(14,165,233,0.15); border: 1px solid rgba(14,165,233,0.3); color:#38bdf8; font-weight:600;">${m}</span>
        `).join('');
      }
    } catch (err) {
      console.warn('Could not load keys status:', err);
    }
  }

  async function saveSettings() {
    const radio = document.querySelector('input[name="pref-ranking-mode"]:checked');
    const ranking_mode = radio ? radio.value : 'best_value';
    const maxBudgetInput = document.getElementById('pref-max-budget');
    const max_budget = maxBudgetInput ? parseFloat(maxBudgetInput.value) : 10;
    const preferFreeInput = document.getElementById('pref-prefer-free');
    const prefer_free = preferFreeInput ? preferFreeInput.checked : true;
    const trustPolicySelect = document.getElementById('pref-trust-policy');
    const trust_policy = trustPolicySelect ? trustPolicySelect.value : 'standard';

    const prefs = { ranking_mode, max_budget, prefer_free, trust_policy };
    localStorage.setItem('syntra_user_preferences', JSON.stringify(prefs));
    state.currentMode = ranking_mode;

    // Sync tab button if currently in studio
    elements.rankTabBtns.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === ranking_mode);
    });

    const ind = document.getElementById('settings-saved-indicator');
    if (ind) {
      ind.classList.remove('hidden');
      setTimeout(() => {
        ind.classList.add('hidden');
      }, 2500);
    }
  }

  async function saveGeminiApiKey() {
    const keyInput = document.getElementById('pref-gemini-key');
    const keyVal = keyInput ? keyInput.value.trim() : '';
    if (!keyVal) {
      alert('Please enter a valid Gemini API key.');
      return;
    }

    try {
      const res = await apiRequest('/api/v1/settings/keys', 'POST', {
        gemini_api_key: keyVal
      });
      alert('Google Gemini API Key successfully saved and connected! Gemini models are now active.');
      if (keyInput) keyInput.value = '';
      await loadSettings();
      triggerRecommendations();
      probeSystemHealth();
    } catch (err) {
      alert(`Failed to save Gemini key: ${err.message}`);
    }
  }

  // =========================================================================
  // Event Listeners Initialization
  // =========================================================================
  function initEventListeners() {
    // Nav tabs
    if (elements.tabStudioBtn) elements.tabStudioBtn.onclick = () => switchView('studio');
    if (elements.tabDashboardBtn) elements.tabDashboardBtn.onclick = () => switchView('dashboard');
    if (elements.tabCatalogueBtn) elements.tabCatalogueBtn.onclick = () => switchView('catalogue');
    if (elements.tabHistoryBtn) elements.tabHistoryBtn.onclick = () => switchView('history');
    if (elements.tabSettingsBtn) elements.tabSettingsBtn.onclick = () => switchView('settings');
    if (elements.saveSettingsBtn) elements.saveSettingsBtn.onclick = saveSettings;

    // Token Card Collapse Toggle
    if (elements.tokenCardHeaderToggle) {
      elements.tokenCardHeaderToggle.onclick = () => {
        if (elements.tokenBody) elements.tokenBody.classList.toggle('hidden');
        if (elements.tokenChevron) {
          elements.tokenChevron.textContent = elements.tokenBody && elements.tokenBody.classList.contains('hidden') ? '▶' : '▼';
        }
      };
    }

    // Prompt Input
    elements.promptInput.oninput = () => {
      updateInputCounter();
      // Debounced understanding
      clearTimeout(state.typingTimeout);
      state.typingTimeout = setTimeout(triggerUnderstanding, 600);
    };

    // Sample Chips
    document.querySelectorAll('.chip-btn').forEach(btn => {
      btn.onclick = (e) => {
        elements.promptInput.value = e.target.getAttribute('data-prompt');
        updateInputCounter();
        triggerUnderstanding();
      };
    });

    // Image Upload
    elements.imageFileInput.onchange = handleImageUpload;
    elements.removeImageBtn.onclick = removeImageAttachment;

    // Voice Recording
    elements.voiceRecordBtn.onclick = toggleAudioRecording;
    elements.stopAudioBtn.onclick = stopAudioRecording;

    // Clear Button
    elements.clearPromptBtn.onclick = () => {
      elements.promptInput.value = '';
      removeImageAttachment();
      updateInputCounter();
    };

    // Execute Main Button
    elements.analyzeRouteBtn.onclick = () => executeRequest(null);

    // Ranking Mode Tabs
    elements.rankTabBtns.forEach(btn => {
      btn.onclick = (e) => {
        const mode = e.target.getAttribute('data-mode');
        updateActiveRankTab(mode);
      };
    });

    // Copy Output Button
    elements.copyOutputBtn.onclick = () => {
      const text = elements.outputContentArea.innerText;
      navigator.clipboard.writeText(text);
      elements.copyOutputBtn.innerHTML = '✓ Copied!';
      setTimeout(() => {
        elements.copyOutputBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy Output
        `;
      }, 2000);
    };

    // Tool Accordion Toggle
    elements.toolAuditToggle.onclick = () => {
      elements.toolAuditContent.classList.toggle('hidden');
    };

    // Catalogue Search & Filters
    elements.catalogueSearchInput.oninput = (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = state.catalogueData.filter(m =>
        m.name.toLowerCase().includes(q) ||
        m.provider.toLowerCase().includes(q) ||
        m.supported_tasks.some(t => t.includes(q))
      );
      renderCatalogueTable(filtered);
    };

    elements.catalogueFilterPills.forEach(pill => {
      pill.onclick = (e) => {
        elements.catalogueFilterPills.forEach(p => p.classList.remove('active'));
        e.target.classList.add('active');
        const filter = e.target.getAttribute('data-filter');
        if (filter === 'free') {
          renderCatalogueTable(state.catalogueData.filter(m => m.is_free));
        } else if (filter === 'paid') {
          renderCatalogueTable(state.catalogueData.filter(m => !m.is_free));
        } else {
          renderCatalogueTable(state.catalogueData);
        }
      };
    });

    // Auth Modal Controls
    elements.authModalClose.onclick = closeAuthModal;
    elements.tabLoginBtn.onclick = () => switchAuthForm('login');
    elements.tabRegisterBtn.onclick = () => switchAuthForm('register');
    elements.forgotPasswordLink.onclick = () => switchAuthForm('reset');
    elements.cancelResetBtn.onclick = () => switchAuthForm('login');

    // Password Visibility Toggles
    document.querySelectorAll('.pwd-toggle-btn').forEach(btn => {
      btn.onclick = (e) => {
        const targetId = e.currentTarget.getAttribute('data-target');
        const input = document.getElementById(targetId);
        if (input.type === 'password') {
          input.type = 'text';
          e.currentTarget.textContent = '🔒';
        } else {
          input.type = 'password';
          e.currentTarget.textContent = '👁️';
        }
      };
    });

    // Context DNA Tab switching
    if (elements.dnaTabs) {
      elements.dnaTabs.forEach(tab => {
        tab.onclick = (e) => {
          elements.dnaTabs.forEach(t => t.classList.remove('active'));
          e.target.classList.add('active');
          activeDnaTab = e.target.getAttribute('data-dna-tab');
          if (state.lastContextDNA) {
            displayDnaTabContent(state.lastContextDNA, activeDnaTab);
          }
        };
      });
    }

    // Context DNA Header Toggle
    if (elements.dnaHeaderToggle) {
      elements.dnaHeaderToggle.onclick = () => {
        elements.dnaBody.classList.toggle('hidden');
      };
    }

    // Post-Response Recovery Actions
    if (elements.btnRetryTenacity) {
      elements.btnRetryTenacity.onclick = async () => {
        if (!state.lastPrompt) return;
        executeRequest(state.lastRecommendations ? state.lastRecommendations.models[0].id : null);
      };
    }

    if (elements.btnImproveOutput) {
      elements.btnImproveOutput.onclick = async () => {
        const notes = prompt("Enter specific improvement instructions (e.g., 'Make it more concise', 'Add unit tests'):");
        if (!notes) return;
        try {
          elements.executionLoadingBox.classList.remove('hidden');
          const res = await apiRequest('/api/v1/feedback', 'POST', {
            session_id: state.lastSessionId || 'sess_feedback',
            prompt: state.lastPrompt || elements.promptInput.value,
            current_output: state.lastOutput || '',
            feedback_type: 'improve',
            notes: notes
          });
          if (res.migration) {
            elements.outputContentArea.innerHTML = formatMarkdown(res.migration.continuation_output);
            if (res.migration.verification_report) renderTrustLayer(res.migration.verification_report);
            if (res.migration.context_dna) renderContextDNA(res.migration.context_dna);
            if (res.migration.execution_trace) renderExecutionTrace(res.migration.execution_trace);
          }
        } catch (err) {
          alert(`Improvement failed: ${err.message}`);
        } finally {
          elements.executionLoadingBox.classList.add('hidden');
        }
      };
    }

    if (elements.btnVerifyTrust) {
      elements.btnVerifyTrust.onclick = async () => {
        if (!state.lastOutput) return;
        try {
          const res = await apiRequest('/api/v1/verify', 'POST', {
            prompt: state.lastPrompt || elements.promptInput.value,
            output: state.lastOutput,
            task_category: state.lastUnderstanding ? state.lastUnderstanding.task_category : 'general',
            policy: 'strict'
          });
          renderTrustLayer(res);
          alert(`Trust Layer re-verification complete: ${res.decision} (${Math.round(res.confidence_score * 100)}% Confidence)`);
        } catch (err) {
          alert(`Verification failed: ${err.message}`);
        }
      };
    }

    if (elements.btnSwitchEngine) {
      elements.btnSwitchEngine.onclick = async () => {
        try {
          elements.executionLoadingBox.classList.remove('hidden');
          const res = await apiRequest('/api/v1/feedback', 'POST', {
            session_id: state.lastSessionId || 'sess_switch',
            prompt: state.lastPrompt || elements.promptInput.value,
            current_output: state.lastOutput || '',
            feedback_type: 'switch_model',
            target_model: 'qwen2.5-coder:7b'
          });
          if (res.migration) {
            elements.outputContentArea.innerHTML = formatMarkdown(res.migration.continuation_output);
            elements.engineUsedTag.textContent = `Engine: ${res.migration.target_model}`;
            if (res.migration.verification_report) renderTrustLayer(res.migration.verification_report);
            if (res.migration.context_dna) renderContextDNA(res.migration.context_dna);
            if (res.migration.execution_trace) renderExecutionTrace(res.migration.execution_trace);
          }
        } catch (err) {
          alert(`Model switch failed: ${err.message}`);
        } finally {
          elements.executionLoadingBox.classList.add('hidden');
        }
      };
    }

    if (elements.btnReportIssue) {
      elements.btnReportIssue.onclick = async () => {
        const issue = prompt("Please describe the issue or defect detected:");
        if (!issue) return;
        await apiRequest('/api/v1/feedback', 'POST', {
          session_id: state.lastSessionId || 'sess_feedback',
          prompt: state.lastPrompt || '',
          current_output: state.lastOutput || '',
          feedback_type: 'report_issue',
          notes: issue
        });
        alert("Thank you! Feedback recorded in canonical audit history.");
      };
    }

    // Jury Demo Modal Controls
    if (elements.navJuryDemoBtn) {
      elements.navJuryDemoBtn.onclick = () => {
        elements.juryDemoModal.classList.remove('hidden');
        elements.juryLiveResults.classList.add('hidden');
      };
    }

    if (elements.juryModalClose) {
      elements.juryModalClose.onclick = () => {
        elements.juryDemoModal.classList.add('hidden');
      };
    }

    // Jury Preset Buttons
    if (elements.juryPresetBtns) {
      elements.juryPresetBtns.forEach(btn => {
        btn.onclick = (e) => {
          elements.juryPresetBtns.forEach(b => b.classList.remove('active'));
          e.target.classList.add('active');
          const preset = e.target.getAttribute('data-preset');
          if (preset === 'coding') {
            elements.juryPromptInput.value = "Implement a robust Python data pipeline with schema validation and summary metrics in JSON format.";
          } else if (preset === 'math') {
            elements.juryPromptInput.value = "Calculate 15 * 15 and explain the steps";
          } else if (preset === 'json') {
            elements.juryPromptInput.value = "Design a REST API spec for an e-commerce order system with JSON request/response bodies";
          }
        };
      });
    }

    // Run Controlled Failover Demo
    if (elements.juryRunFailoverBtn) {
      elements.juryRunFailoverBtn.onclick = async () => {
        const promptText = elements.juryPromptInput.value.trim();
        const sourceModel = elements.jurySourceModel.value;
        const simulatedFailure = elements.juryFailureType.value;
        const preferFree = elements.juryFreeOnlyToggle.checked;

        elements.juryLiveResults.classList.remove('hidden');
        elements.juryResultsTitle.textContent = "Injecting failure & testing failover loop...";
        elements.juryStatusDot.className = "status-indicator-dot running";

        for (let i = 1; i <= 8; i++) {
          const el = document.getElementById(`jstep-${i}`);
          if (el) el.className = "jury-step-card";
        }

        try {
          const res = await apiRequest('/api/v1/demo/simulate-failure-migration', 'POST', {
            prompt: promptText,
            source_model: sourceModel,
            simulated_failure: simulatedFailure,
            prefer_free_only: preferFree
          });

          const stepMeta = [
            `Classified intent`,
            `Simulated [${res.failure_category || 'QUOTA'}]`,
            `Tenacity policy bounded`,
            `Goal & artifacts preserved`,
            `Rerouted to ${res.target_provider}`,
            `${res.compression_ratio}x compressed`,
            `Executed on ${res.target_model}`,
            `${res.verification_report.decision} (${Math.round(res.verification_report.confidence_score * 100)}%)`
          ];

          for (let i = 1; i <= 8; i++) {
            const el = document.getElementById(`jstep-${i}`);
            const metaEl = document.getElementById(`jstep-${i}-meta`);
            if (el) el.className = "jury-step-card completed";
            if (metaEl) metaEl.textContent = stepMeta[i - 1];
          }

          elements.juryStatusDot.className = "status-indicator-dot";
          elements.juryResultsTitle.textContent = `Failover Handover Succeeded: Active Model is now ${res.target_model}`;
          elements.juryResultsLatency.textContent = `${res.total_latency_ms} ms`;

          elements.juryDnaPreview.textContent = JSON.stringify(res.context_dna, null, 2);
          elements.juryOutputPreview.textContent = res.continuation_output;

        } catch (err) {
          elements.juryStatusDot.className = "status-indicator-dot failed";
          elements.juryResultsTitle.textContent = `Demonstration error: ${err.message}`;
        }
      };
    }

    // Settings & API Key Handlers
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    if (saveSettingsBtn) {
      saveSettingsBtn.onclick = saveSettings;
    }

    const saveGeminiKeyBtn = document.getElementById('save-gemini-key-btn');
    if (saveGeminiKeyBtn) {
      saveGeminiKeyBtn.onclick = saveGeminiApiKey;
    }

    // Form Submissions
    elements.loginForm.onsubmit = async (e) => {
      e.preventDefault();
      elements.authErrorBanner.classList.add('hidden');
      try {
        const res = await apiRequest('/api/v1/auth/login', 'POST', {
          email: document.getElementById('login-email').value,
          password: document.getElementById('login-password').value
        });
        state.token = res.access_token;
        localStorage.setItem('syntra_token', res.access_token);
        state.user = res;
        renderAuthenticatedControls();
        closeAuthModal();
      } catch (err) {
        showAuthError(err.message);
      }
    };

    elements.registerForm.onsubmit = async (e) => {
      e.preventDefault();
      elements.authErrorBanner.classList.add('hidden');
      try {
        const res = await apiRequest('/api/v1/auth/register', 'POST', {
          name: document.getElementById('reg-name').value,
          email: document.getElementById('reg-email').value,
          password: document.getElementById('reg-password').value
        });
        state.token = res.access_token;
        localStorage.setItem('syntra_token', res.access_token);
        state.user = res;
        renderAuthenticatedControls();
        closeAuthModal();
      } catch (err) {
        showAuthError(err.message);
      }
    };

    elements.resetForm.onsubmit = async (e) => {
      e.preventDefault();
      elements.authErrorBanner.classList.add('hidden');
      try {
        const res = await apiRequest('/api/v1/auth/reset-password', 'POST', {
          email: document.getElementById('reset-email').value,
          new_password: document.getElementById('reset-new-password').value
        });
        alert(res.message);
        switchAuthForm('login');
      } catch (err) {
        showAuthError(err.message);
      }
    };

    elements.launchDemoBtn.onclick = async () => {
      try {
        const res = await apiRequest('/api/v1/auth/demo', 'POST');
        state.token = res.access_token;
        localStorage.setItem('syntra_token', res.access_token);
        state.user = res;
        renderAuthenticatedControls();
        closeAuthModal();
      } catch (err) {
        showAuthError(err.message);
      }
    };
  }

  // =========================================================================
  // Initialize App
  // =========================================================================
  window.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    checkAuthSession();
    probeSystemHealth();
    triggerRecommendations();
  });

})();
