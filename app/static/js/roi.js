(() => {
  const input = document.getElementById('video');
  const canvas = document.getElementById('roi-canvas');
  const section = document.getElementById('zone-config');
  if (!input || !canvas || !section) return;

  const ctx = canvas.getContext('2d');
  const stage = section.querySelector('.zone-stage-wrap');
  const valueInput = document.getElementById('roi-zones');
  const video = document.getElementById('preview');
  const hint = document.getElementById('zone-hint');
  const frameStatus = document.getElementById('zone-frame-status');
  const frameControl = document.getElementById('zone-frame-control');
  const frameSlider = document.getElementById('zone-frame-slider');
  const frameTime = document.getElementById('zone-frame-time');
  const nameInput = document.getElementById('zone-name');
  const enabledInput = document.getElementById('zone-enabled');
  const drawButton = document.getElementById('zone-draw');
  const finishButton = document.getElementById('zone-finish');
  const undoButton = document.getElementById('zone-undo');
  const clearButton = document.getElementById('zone-clear');
  const list = document.getElementById('zone-list');

  let sourceFrame = null;
  let sampleCanvas = null;
  let zones = [];
  let points = [];
  let drawing = false;
  let cursor = null;
  let frameReady = false;
  let resizeObserver = null;
  let loadToken = 0;

  function setFrameStatus(message, state = 'ready') {
    frameStatus.textContent = message;
    frameStatus.dataset.state = state;
  }

  function formatClock(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '00:00.0';
    const whole = Math.floor(seconds);
    const minutes = Math.floor(whole / 60);
    const remainder = (seconds - minutes * 60).toFixed(1).padStart(4, '0');
    return `${String(minutes).padStart(2, '0')}:${remainder}`;
  }

  function setHint(message) {
    hint.textContent = message;
  }

  function statusForPoints() {
    if (!frameReady) return;
    if (!drawing) return;
    if (points.length < 3) {
      setHint(`${points.length}/3 points selected. Add ${3 - points.length} more point${points.length === 2 ? '' : 's'} to finish.`);
    } else {
      setHint('Polygon ready to finish. Click Finish Polygon or click near point 1 to close it.');
    }
  }

  function persist() {
    valueInput.value = JSON.stringify(zones.map(zone => ({
      zone_id: zone.zone_id,
      name: zone.name,
      enabled: zone.enabled,
      points: zone.points.map(point => [point[0], point[1]])
    })));
    list.replaceChildren();
    zones.forEach((zone, index) => {
      const item = document.createElement('li');
      item.textContent = `Zone ${index + 1}: ${zone.name} | ${zone.enabled ? 'Enabled' : 'Disabled'} | ${zone.points.length} points | Saved`;
      list.append(item);
    });
    clearButton.disabled = zones.length === 0 && points.length === 0;
    undoButton.disabled = points.length === 0 && zones.length === 0;
    finishButton.disabled = !drawing || points.length < 3;
    drawButton.disabled = drawing || !frameReady;
    canvas.classList.toggle('is-drawing', drawing);
    render();
  }

  function toCanvas(point) {
    return [point[0] * canvas.width, point[1] * canvas.height];
  }

  function drawMarker(point, index, active, first = false) {
    const [x, y] = toCanvas(point);
    const radius = Math.max(12, Math.min(canvas.width, canvas.height) / 36);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = active ? '#ff4d61' : '#56d9ef';
    ctx.fill();
    ctx.lineWidth = Math.max(2, radius / 7);
    ctx.strokeStyle = first && drawing ? '#ffffff' : '#071421';
    ctx.stroke();
    ctx.fillStyle = '#071421';
    ctx.font = `700 ${Math.max(12, radius * 1.1)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(index + 1), x, y + 0.5);
  }

  function drawPolygon(pointsToDraw, active, complete, showNumbers = true) {
    if (!pointsToDraw.length) return;
    const coords = pointsToDraw.map(toCanvas);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = Math.max(3, canvas.width / 360);
    if (complete || coords.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(coords[0][0], coords[0][1]);
      coords.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
      ctx.closePath();
      ctx.fillStyle = active ? 'rgba(255, 61, 82, .30)' : 'rgba(86, 217, 239, .18)';
      ctx.fill();
    }
    ctx.beginPath();
    ctx.moveTo(coords[0][0], coords[0][1]);
    coords.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    if (complete) ctx.closePath();
    ctx.strokeStyle = active ? '#ff4d61' : '#56d9ef';
    ctx.setLineDash(complete ? [] : [canvas.width / 160, canvas.width / 240]);
    ctx.stroke();
    ctx.setLineDash([]);
    if (showNumbers) pointsToDraw.forEach((point, index) => drawMarker(point, index, active, index === 0));
  }

  function render() {
    if (!frameReady || !sourceFrame) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(sourceFrame, 0, 0, canvas.width, canvas.height);
    zones.forEach(zone => drawPolygon(zone.points, zone.enabled, true));
    if (points.length) {
      drawPolygon(points, enabledInput.checked, false);
      if (drawing && cursor) {
        const last = toCanvas(points[points.length - 1]);
        const next = toCanvas(cursor);
        ctx.beginPath();
        ctx.moveTo(last[0], last[1]);
        ctx.lineTo(next[0], next[1]);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = Math.max(2, canvas.width / 500);
        ctx.setLineDash([canvas.width / 120, canvas.width / 180]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  function resizeCanvas() {
    if (!frameReady || !canvas.width || !canvas.height) return;
    const availableWidth = Math.max(1, stage.clientWidth - 22);
    const maxHeight = Math.min(460, Math.max(220, window.innerHeight * 0.58));
    const scale = Math.min(1, availableWidth / canvas.width, maxHeight / canvas.height);
    canvas.style.width = `${Math.max(1, canvas.width * scale)}px`;
    canvas.style.height = `${Math.max(1, canvas.height * scale)}px`;
    render();
  }

  function looksLikeBlackFrame() {
    if (!sampleCanvas) {
      sampleCanvas = document.createElement('canvas');
      sampleCanvas.width = 24;
      sampleCanvas.height = 16;
    }
    const sampleContext = sampleCanvas.getContext('2d', { willReadFrequently: true });
    sampleContext.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
    const pixels = sampleContext.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data;
    let total = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      total += (pixels[index] * 0.2126 + pixels[index + 1] * 0.7152 + pixels[index + 2] * 0.0722);
    }
    return total / (pixels.length / 4) < 8;
  }

  function captureFrame(token = loadToken) {
    if (token !== loadToken || !video.videoWidth || !video.videoHeight || video.readyState < 2) return false;
    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      sourceFrame = document.createElement('canvas');
      sourceFrame.width = canvas.width;
      sourceFrame.height = canvas.height;
      sourceFrame.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      frameReady = true;
      canvas.hidden = false;
      stage.classList.remove('is-loading');
      stage.hidden = false;
      drawButton.disabled = false;
      frameSlider.disabled = false;
      frameSlider.value = Number.isFinite(video.currentTime) ? video.currentTime : 0;
      frameTime.value = formatClock(Number(frameSlider.value));
      resizeCanvas();
      if (looksLikeBlackFrame()) {
        setFrameStatus('This decoded frame is very dark. Use the frame slider to choose a clearer moment.', 'ready');
      } else {
        setFrameStatus(`Video frame ready - ${canvas.width} x ${canvas.height}`, 'ready');
      }
      if (!zones.length && !drawing) setHint('Video frame ready. Draw a zone if needed, or continue without one.');
      persist();
      return true;
    } catch (_error) {
      frameReady = false;
      canvas.hidden = true;
      setFrameStatus('The selected frame could not be drawn. Try another frame or a different video.', 'error');
      setHint('Frame decoding failed; polygon coordinates were not changed.');
      return false;
    }
  }

  function onFrameDecoded() {
    if (video.readyState >= 2) captureFrame();
  }

  video.addEventListener('loadedmetadata', updateFrameDuration);
  video.addEventListener('loadeddata', onFrameDecoded);
  video.addEventListener('seeked', () => {
    frameSlider.value = String(video.currentTime);
    frameTime.value = formatClock(video.currentTime);
    captureFrame();
  });
  video.addEventListener('error', () => {
    if (!input.files?.[0]) return;
    frameReady = false;
    stage.classList.remove('is-loading');
    canvas.hidden = true;
    setFrameStatus('This video could not be decoded by the browser. Try another frame or a supported MP4, AVI, MOV, or MKV file.', 'error');
    setHint('The preview could not be prepared. The upload form remains available, but zones cannot be drawn until a frame decodes.');
    drawButton.disabled = true;
  });

  function updateFrameDuration() {
    const duration = Number(video.duration);
    if (!Number.isFinite(duration) || duration <= 0) {
      frameSlider.disabled = true;
      setFrameStatus('The video opened, but its timeline duration is unavailable. You can still draw on the decoded frame.', 'ready');
      return;
    }
    frameSlider.max = String(duration);
    frameSlider.step = '0.1';
    frameSlider.value = '0';
    frameControl.hidden = false;
    frameSlider.disabled = false;
    frameTime.value = formatClock(0);
    if (video.readyState >= 2) captureFrame();
  }

  input.addEventListener('change', () => {
    const token = ++loadToken;
    zones = [];
    points = [];
    cursor = null;
    drawing = false;
    frameReady = false;
    sourceFrame = null;
    canvas.hidden = true;
    valueInput.value = '[]';
    stage.hidden = false;
    stage.classList.add('is-loading');
    frameControl.hidden = true;
    frameSlider.disabled = true;
    drawButton.disabled = true;
    canvas.classList.remove('is-drawing');
    setFrameStatus('Preparing the first decoded frame...', 'loading');
    setHint('Loading a decoded frame from the selected video...');
    if (!input.files?.[0]) {
      stage.hidden = true;
      stage.classList.remove('is-loading');
      persist();
      return;
    }
    persist();
    if (video.readyState >= 2) {
      updateFrameDuration();
      captureFrame(token);
    }
  });

  frameSlider.addEventListener('input', () => {
    const seconds = Number(frameSlider.value);
    frameTime.value = formatClock(seconds);
    if (!Number.isFinite(seconds) || video.readyState < 1) return;
    drawButton.disabled = true;
    setFrameStatus(`Seeking to ${formatClock(seconds)}...`, 'loading');
    try {
      video.currentTime = seconds;
      if (Math.abs(video.currentTime - seconds) < 0.001 && video.readyState >= 2) captureFrame();
    } catch (_error) {
      drawButton.disabled = true;
      setFrameStatus('The browser could not seek to that frame. Try a nearby time.', 'error');
    }
  });
  drawButton.addEventListener('click', () => {
    if (!frameReady) {
      setHint('Select a supported video and wait until Video frame ready appears before drawing.');
      return;
    }
    drawing = true;
    points = [];
    cursor = null;
    setFrameStatus(`Drawing zone ${zones.length + 1}`, 'ready');
    setHint('0/3 points selected. Click at least three points around the restricted area.');
    persist();
  });

  function normalizedPoint(event) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return [
      Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height))
    ];
  }

  canvas.addEventListener('pointermove', event => {
    if (!drawing) return;
    cursor = normalizedPoint(event);
    render();
  });
  canvas.addEventListener('pointerleave', () => { cursor = null; render(); });
  canvas.addEventListener('click', event => {
    if (!drawing || !frameReady) return;
    const point = normalizedPoint(event);
    if (!point) return;
    const rect = canvas.getBoundingClientRect();
    if (points.length >= 3) {
      const dx = (point[0] - points[0][0]) * rect.width;
      const dy = (point[1] - points[0][1]) * rect.height;
      if (Math.hypot(dx, dy) <= 18) {
        finishZone();
        return;
      }
    }
    points.push([Number(point[0].toFixed(6)), Number(point[1].toFixed(6))]);
    cursor = null;
    statusForPoints();
    persist();
  });

  function finishZone() {
    if (!drawing || points.length < 3) {
      setHint('A zone needs at least three points before it can be finished.');
      return;
    }
    const index = zones.length + 1;
    zones.push({
      zone_id: `zone_${index}`,
      name: nameInput.value.trim() || `Zone ${index}`,
      enabled: enabledInput.checked,
      points: points.map(point => [...point])
    });
    points = [];
    cursor = null;
    drawing = false;
    setFrameStatus('Video frame ready | zone saved', 'ready');
    setHint(`Zone saved. Choose a name and click Draw Zone to add another polygon.`);
    persist();
  }

  finishButton.addEventListener('click', finishZone);
  undoButton.addEventListener('click', () => {
    if (points.length) {
      points.pop();
      statusForPoints();
    } else if (zones.length) {
      const last = zones.pop();
      points = last.points.slice(0, -1);
      drawing = true;
      cursor = null;
      nameInput.value = last.name;
      enabledInput.checked = last.enabled;
      setFrameStatus(`Drawing zone ${zones.length + 1}`, 'ready');
      statusForPoints();
    }
    persist();
  });
  clearButton.addEventListener('click', () => {
    zones = [];
    points = [];
    drawing = false;
    cursor = null;
    setFrameStatus(frameReady ? 'Video frame ready' : 'Preparing video frame...', frameReady ? 'ready' : 'loading');
    setHint('All zones cleared. Video upload remains available without a zone.');
    persist();
  });
  enabledInput.addEventListener('change', () => {
    zones.forEach(zone => { zone.enabled = enabledInput.checked; });
    persist();
  });

  if ('ResizeObserver' in window) {
    resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(stage);
  } else {
    window.addEventListener('resize', resizeCanvas);
  }
  persist();
})();
