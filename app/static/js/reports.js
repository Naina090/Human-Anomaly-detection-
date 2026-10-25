(() => {
  const root = document.querySelector('.reports-page');
  if (!root) return;

  const tabs = [...root.querySelectorAll('[role="tab"][data-report-tab]')];
  const panels = [...root.querySelectorAll('[role="tabpanel"]')];
  const setTab = (name, updateUrl = true) => {
    if (!tabs.some(tab => tab.dataset.reportTab === name)) name = 'overview';
    root.dataset.activeTab = name;
    tabs.forEach(tab => {
      const active = tab.dataset.reportTab === name;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach(panel => { panel.hidden = panel.id !== `panel-${name}`; });
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set('tab', name);
      window.history.replaceState({}, '', url);
    }
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => setTab(tab.dataset.reportTab));
    tab.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault();
      tabs[next].focus();
      setTab(tabs[next].dataset.reportTab);
    });
  });
  root.querySelectorAll('[data-open-tab]').forEach(button => button.addEventListener('click', () => {
    const target = tabs.find(tab => tab.dataset.reportTab === button.dataset.openTab);
    if (target) { setTab(target.dataset.reportTab); target.focus(); }
  }));

  root.querySelectorAll('[data-expand-report]').forEach(button => button.addEventListener('click', () => {
    const row = document.getElementById(button.getAttribute('aria-controls'));
    if (!row) return;
    const opening = row.hidden;
    root.querySelectorAll('.report-detail-row').forEach(other => { other.hidden = true; });
    root.querySelectorAll('[data-expand-report]').forEach(other => {
      other.setAttribute('aria-expanded', 'false');
      other.textContent = 'View Report';
    });
    row.hidden = !opening;
    button.setAttribute('aria-expanded', String(opening));
    button.textContent = opening ? 'Close Report' : 'View Report';
  }));

  setTab(root.dataset.activeTab || 'overview', false);
})();
