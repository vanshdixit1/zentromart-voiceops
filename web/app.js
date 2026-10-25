(() => {
  const $ = (selector) => document.querySelector(selector);
  const conversation = $('#conversation');
  const form = $('#chatForm');
  const input = $('#messageInput');
  const sendButton = $('#sendButton');
  const recordButton = $('#recordButton');
  const stepSelect = $('#stepSelect');
  const recordingStatus = $('#recordingStatus');
  const recordingLabel = $('#recordingLabel');
  const toast = $('#toast');
  let sessionId = '';
  let sessionState = '';
  let voiceReady = false;
  let busy = false;
  let recording = false;
  let audioContext = null;
  let micStream = null;
  let processor = null;
  let silentGain = null;
  let audioChunks = [];
  let recordStartedAt = 0;
  let recordTimer = null;
  let toastTimer = null;

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 4300);
  }

  async function request(path, payload) {
    const requestPayload = { ...(payload || {}) };
    if (requestPayload.session_id && sessionState) requestPayload.session_state = sessionState;
    let response;
    try {
      response = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
      });
    } catch (_) {
      throw new Error('Could not reach the VoiceOps server. Check your connection and try again.');
    }
    let data;
    try { data = await response.json(); } catch (_) { data = {}; }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    if (typeof data.session_state === 'string') sessionState = data.session_state;
    return data;
  }

  function setBusy(value) {
    busy = value;
    sendButton.disabled = value;
    stepSelect.disabled = value;
    recordButton.disabled = value || !voiceReady || stepSelect.value !== '3';
    input.disabled = value;
  }

  async function createSession() {
    sessionState = '';
    const data = await request('/api/session', { step: Number(stepSelect.value) });
    sessionId = data.session_id;
    sessionState = data.session_state || '';
    $('#welcomeCard').hidden = false;
    conversation.querySelectorAll('.message-row').forEach((row) => row.remove());
    conversation.scrollTop = 0;
    input.focus();
  }

  function addUserMessage(text, isVoice = false) {
    $('#welcomeCard').hidden = true;
    const row = document.createElement('div');
    row.className = 'message-row user';
    const content = document.createElement('div');
    content.className = 'message-content';
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = isVoice ? 'Your voice · transcript' : 'You';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (isVoice) {
      const label = document.createElement('span');
      label.className = 'transcript-label';
      label.textContent = 'Gnani transcript';
      bubble.append(label);
    }
    const textNode = document.createElement('span');
    textNode.textContent = text;
    bubble.append(textNode);
    content.append(meta, bubble);
    row.append(content);
    conversation.append(row);
    conversation.scrollTop = conversation.scrollHeight;
    return { row, textNode, bubble };
  }

  function addTyping() {
    const row = document.createElement('div');
    row.className = 'message-row';
    row.dataset.typing = 'true';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'z';
    const bubble = document.createElement('div');
    bubble.className = 'bubble typing';
    for (let i = 0; i < 3; i += 1) bubble.append(document.createElement('i'));
    row.append(avatar, bubble);
    conversation.append(row);
    conversation.scrollTop = conversation.scrollHeight;
    return row;
  }

  function playAudio(encoded, row) {
    if (!encoded) return;
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    const audio = document.createElement('audio');
    audio.className = 'reply-audio';
    audio.controls = true;
    audio.preload = 'none';
    audio.src = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
    row.append(audio);
  }

  function addAssistantMessage(result) {
    const row = document.createElement('div');
    row.className = 'message-row';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'z';
    const content = document.createElement('div');
    content.className = 'message-content';
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = 'ZentroMart assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = result.answer || 'I could not prepare a reply. Please try again.';
    content.append(meta, bubble);

    if (Array.isArray(result.sources) && result.sources.length) {
      const sources = document.createElement('div');
      sources.className = 'sources';
      sources.setAttribute('aria-label', 'Retrieved sources');
      result.sources.forEach((source) => {
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        chip.textContent = source;
        sources.append(chip);
      });
      content.append(sources);
    }

    (result.tool_events || []).forEach((event) => {
      let label = '';
      if (event.action === 'lookup_order' && event.verified) label = 'Order identity verified';
      else if (event.action === 'lookup_order' && event.verification_required) label = 'Order verification required';
      else if (event.action === 'create_ticket') label = event.created ? 'Support ticket confirmed' : 'Ticket could not be created';
      if (label) {
        const note = document.createElement('p');
        note.className = 'tool-note';
        note.textContent = `✓ ${label}`;
        content.append(note);
      }
    });

    row.append(avatar, content);
    conversation.append(row);
    playAudio(result.audio_base64, content);
    conversation.scrollTop = conversation.scrollHeight;
  }

  async function sendText(message) {
    if (busy || !message.trim()) return;
    const value = message.trim();
    addUserMessage(value);
    input.value = '';
    input.style.height = 'auto';
    setBusy(true);
    const typing = addTyping();
    try {
      const result = await request('/api/chat', {
        session_id: sessionId,
        message: value,
        speak: $('#speakReplies').checked,
      });
      typing.remove();
      addAssistantMessage(result);
    } catch (error) {
      typing.remove();
      showToast(error.message);
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  function encodeWav(samples, sampleRate) {
    const targetRate = 16000;
    const ratio = sampleRate / targetRate;
    const outLength = Math.max(1, Math.floor(samples.length / ratio));
    const pcm = new Int16Array(outLength);
    for (let i = 0; i < outLength; i += 1) {
      const sourceIndex = i * ratio;
      const left = Math.floor(sourceIndex);
      const right = Math.min(left + 1, samples.length - 1);
      const fraction = sourceIndex - left;
      const value = samples[left] * (1 - fraction) + samples[right] * fraction;
      pcm[i] = Math.max(-1, Math.min(1, value)) * (value < 0 ? 32768 : 32767);
    }
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);
    const writeText = (offset, text) => {
      for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
    };
    writeText(0, 'RIFF'); view.setUint32(4, 36 + pcm.length * 2, true); writeText(8, 'WAVE');
    writeText(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); writeText(36, 'data'); view.setUint32(40, pcm.length * 2, true);
    new Int16Array(buffer, 44).set(pcm);
    return buffer;
  }

  function encodeBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    const block = 0x8000;
    for (let i = 0; i < bytes.length; i += block) {
      binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + block, bytes.length)));
    }
    return btoa(binary);
  }

  async function finishRecording(send) {
    if (!recording) return;
    recording = false;
    clearInterval(recordTimer);
    recordingStatus.hidden = true;
    recordButton.innerHTML = '<span class="mic-icon" aria-hidden="true">●</span><span class="record-label">Record a question</span>';
    recordButton.classList.remove('is-recording');
    if (processor) { processor.disconnect(); processor.onaudioprocess = null; }
    if (silentGain) silentGain.disconnect();
    if (micStream) micStream.getTracks().forEach((track) => track.stop());
    const context = audioContext;
    if (context) await context.close().catch(() => {});
    audioContext = null; processor = null; silentGain = null; micStream = null;
    const chunks = audioChunks;
    audioChunks = [];
    if (!send || !chunks.length) return;

    const sampleRate = context ? context.sampleRate : 48000;
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const samples = new Float32Array(length);
    let offset = 0;
    chunks.forEach((chunk) => { samples.set(chunk, offset); offset += chunk.length; });
    const wav = encodeWav(samples, sampleRate);
    const placeholder = addUserMessage('Sending your recording to Gnani…', true);
    setBusy(true);
    const typing = addTyping();
    try {
      const result = await request('/api/voice', {
        session_id: sessionId,
        audio_base64: encodeBase64(wav),
      });
      placeholder.textNode.textContent = result.transcript || 'No speech detected';
      typing.remove();
      addAssistantMessage(result);
    } catch (error) {
      placeholder.textNode.textContent = 'Voice message was not processed.';
      typing.remove();
      showToast(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function startRecording() {
    if (!voiceReady) {
      showToast('Voice is unavailable. Check that GNANI_API_KEY is configured in .env.');
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast('Microphone access is unavailable. Open the website at its localhost address in Edge or Chrome.');
      return;
    }
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.resume();
      const source = audioContext.createMediaStreamSource(micStream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      audioChunks = [];
      processor.onaudioprocess = (event) => audioChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);
      recordStartedAt = Date.now();
      recording = true;
      recordingStatus.hidden = false;
      recordButton.innerHTML = '<span class="recording-pulse"></span><span class="record-label">Stop & send</span>';
      recordButton.classList.add('is-recording');
      recordTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordStartedAt) / 1000);
        recordingLabel.textContent = `Recording · ${elapsed}s / 60s`;
        if (elapsed >= 60) finishRecording(true);
      }, 250);
    } catch (error) {
      if (micStream) micStream.getTracks().forEach((track) => track.stop());
      showToast(error.name === 'NotAllowedError'
        ? 'Microphone access was blocked. Allow it in your browser’s address-bar permissions and try again.'
        : `Could not start the microphone: ${error.message}`);
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    sendText(input.value);
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendText(input.value);
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
  });
  recordButton.addEventListener('click', () => recording ? finishRecording(true) : startRecording());
  $('#cancelRecording').addEventListener('click', () => finishRecording(false));
  $('#newChat').addEventListener('click', async () => {
    if (busy || recording) return;
    try { await createSession(); } catch (error) { showToast(error.message); }
  });
  stepSelect.addEventListener('change', async () => {
    if (busy || recording) return;
    try {
      await createSession();
      recordButton.disabled = !voiceReady || stepSelect.value !== '3';
      if (stepSelect.value !== '3') {
        showToast('Microphone conversations use Step 3. Steps 1 and 2 are available for typed evaluation.');
      }
    } catch (error) { showToast(error.message); }
  });
  $('#helpButton').addEventListener('click', () => {
    const strip = $('#helpStrip');
    strip.hidden = !strip.hidden;
  });
  document.querySelectorAll('[data-prompt]').forEach((button) => {
    button.addEventListener('click', () => sendText(button.dataset.prompt));
  });
  window.addEventListener('beforeunload', () => {
    if (micStream) micStream.getTracks().forEach((track) => track.stop());
  });

  async function initialize() {
    try {
      const health = await fetch('/api/health').then((response) => response.json());
      if (health.hosted) {
        const pill = document.querySelector('.local-pill');
        if (pill) pill.lastChild.textContent = ' Hosted workspace';
      }
      $('#datasetStatus').textContent = health.dataset_ready ? 'Loaded' : 'Missing';
      $('#datasetStatus').className = `system-value ${health.dataset_ready ? 'ready' : 'offline'}`;
      voiceReady = Boolean(health.voice_ready);
      $('#voiceStatus').textContent = voiceReady ? 'Gnani ready' : 'Key needed';
      $('#voiceStatus').className = `system-value ${voiceReady ? 'ready' : 'offline'}`;
      $('#modelName').textContent = health.model || 'Gemini';
      recordButton.disabled = !voiceReady || stepSelect.value !== '3';
      if (!voiceReady) recordButton.title = 'Set GNANI_API_KEY in .env to enable voice';
      await createSession();
    } catch (error) {
      showToast(error.message || 'Could not connect to the VoiceOps server.');
      $('#datasetStatus').textContent = 'Offline';
      $('#datasetStatus').className = 'system-value offline';
    }
  }
  initialize();
})();
