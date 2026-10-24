(() => {
  const input = document.getElementById('video');
  const canvas = document.getElementById('roi-canvas');
  const section = document.getElementById('zone-config');
  if (!input || !canvas || !section) return;

  const ctx = canvas.getContext('2d');
  const stage = section.querySelector('.zone-stage-wrap');
  const formValue = document.getElementById('roi-zones');
  const video = document.getElementById('preview');
  const hint = document.getElementById('zone-hint');
  const nameInput = document.getElementById('zone-name');
  const enabledInput = document.getElementById('zone-enabled');
  const drawButton = document.getElementById('zone-draw');
  const finishButton = document.getElementById('zone-finish');
  const undoButton = document.getElementById('zone-undo');
  const clearButton = document.getElementById('zone-clear');
  const list = document.getElementById('zone-list');
  let baseFrame = null;
  let zones = [];
  let points = [];
  let drawing = false;

  function persist() {
    formValue.value = JSON.stringify(zones);
    list.replaceChildren();
    zones.forEach((zone, index) => {
      const item = document.createElement('li');
      item.textContent = `${zone.name} · ${zone.enabled ? 'Enabled' : 'Disabled'} · ${zone.points.length} points`;
      list.append(item);
    });
    clearButton.disabled = zones.length === 0 && points.length === 0;
    undoButton.disabled = points.length === 0 && zones.length === 0;
    finishButton.disabled = !drawing || points.length < 3;
    drawButton.disabled = drawing;
    render();
  }

  function render() {
    if (!baseFrame) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(baseFrame, 0, 0);
    const paint = (polygon, active, close) => {
      if (!polygon.length) return;
      ctx.beginPath();
      ctx.moveTo(polygon[0][0] * canvas.width, polygon[0][1] * canvas.height);
      polygon.slice(1).forEach(point => ctx.lineTo(point[0] * canvas.width, point[1] * canvas.height));
      if (close && polygon.length >= 3) ctx.closePath();
      ctx.lineWidth = Math.max(2, canvas.width / 500);
      ctx.strokeStyle = active ? '#ff555e' : '#c1cbd5';
      ctx.fillStyle = active ? 'rgba(240, 65, 75, .22)' : 'rgba(145, 160, 175, .12)';
      if (close && polygon.length >= 3) ctx.fill();
      ctx.stroke();
      polygon.forEach((point, index) => {
        ctx.beginPath();
        ctx.arc(point[0] * canvas.width, point[1] * canvas.height, Math.max(4, canvas.width / 180), 0, Math.PI * 2);
        ctx.fillStyle = index === 0 && drawing ? '#fff' : (active ? '#ff555e' : '#c1cbd5');
        ctx.fill();
      });
    };
    zones.forEach(zone => paint(zone.points, zone.enabled, true));
    paint(points, enabledInput.checked, false);
  }

  function captureFrame() {
    if (!video.videoWidth || !video.videoHeight) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    baseFrame = document.createElement('canvas');
    baseFrame.width = canvas.width;
    baseFrame.height = canvas.height;
    baseFrame.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    stage.hidden = false;
    const resize = () => {
      const ratio = canvas.width / canvas.height;
      const maxHeight = Math.min(420, Math.max(220, window.innerHeight * .55));
      canvas.style.width = `${Math.min(stage.clientWidth, maxHeight * ratio)}px`;
      canvas.style.height = 'auto';
    };
    resize();
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(stage);
    else window.addEventListener('resize', resize);
    render();
  }

  input.addEventListener('change', () => {
    if (!input.files?.[0]) return;
    zones = [];
    points = [];
    drawing = false;
    formValue.value = '[]';
    stage.hidden = true;
    baseFrame = null;
    persist();
    video.addEventListener('loadeddata', captureFrame, { once: true });
    if (video.readyState >= 2) captureFrame();
  });

  drawButton.addEventListener('click', () => {
    if (!baseFrame) {
      hint.textContent = 'Select a video and wait for its first frame to load before drawing.';
      return;
    }
    drawing = true;
    points = [];
    hint.textContent = 'Click at least three points. Click near the first point or choose Finish Polygon.';
    persist();
  });

  canvas.addEventListener('click', event => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
    if (points.length >= 3) {
      const dx = (x - points[0][0]) * rect.width;
      const dy = (y - points[0][1]) * rect.height;
      if (Math.hypot(dx, dy) <= 14) {
        finishZone();
        return;
      }
    }
    points.push([Number(x.toFixed(6)), Number(y.toFixed(6))]);
    hint.textContent = `${points.length} point${points.length === 1 ? '' : 's'} added${points.length < 3 ? ' · add at least 3' : ''}.`;
    persist();
  });

  function finishZone() {
    if (points.length < 3) {
      hint.textContent = 'A zone needs at least three points.';
      return;
    }
    zones.push({ zone_id: `zone_${zones.length + 1}`, name: nameInput.value.trim() || `Zone ${zones.length + 1}`,
      enabled: enabledInput.checked, points });
    points = [];
    drawing = false;
    hint.textContent = 'Zone saved for this upload. Use Draw Zone to add another polygon.';
    persist();
  }

  finishButton.addEventListener('click', finishZone);
  undoButton.addEventListener('click', () => {
    if (points.length) points.pop();
    else if (zones.length) {
      const last = zones.pop();
      points = last.points.slice(0, -1);
      drawing = true;
      nameInput.value = last.name;
      enabledInput.checked = last.enabled;
    }
    hint.textContent = 'Last point removed.';
    persist();
  });
  clearButton.addEventListener('click', () => {
    zones = [];
    points = [];
    drawing = false;
    hint.textContent = 'All zones cleared. Video upload remains available without a zone.';
    persist();
  });
  enabledInput.addEventListener('change', () => {
    zones.forEach(zone => { zone.enabled = enabledInput.checked; });
    persist();
  });
  persist();
})();
