(() => {
  // The mode is system, light or dark, stored under main's key; each scheme shows the theme picked for it: light or dark, or a
  // theme of that scheme from management/themes.py, whose <style> names its scheme. Without a pick, the script tag names the
  // operator's default theme of each scheme.
  const root = document.documentElement;
  const defaults = document.currentScript?.dataset ?? {};
  const storageKeys = { mode: 'selectedTheme', 'light-theme': 'lightTheme', 'dark-theme': 'darkTheme' };
  const prefersDark = matchMedia('(prefers-color-scheme: dark)');

  function stored(name) {
    try { return localStorage.getItem(storageKeys[name]); } catch { return null; }
  }
  function store(name, value) {
    try { localStorage.setItem(storageKeys[name], value); } catch { }
  }
  function themeMode() {
    const mode = stored('mode');
    return mode === 'light' || mode === 'dark' ? mode : 'system';
  }
  function schemeTheme(scheme) {
    const id = stored(`${scheme}-theme`);
    const offered = id === scheme || (id && document.querySelector(`style[data-theme-id="${CSS.escape(id)}"]`)?.dataset.scheme === scheme);
    return offered ? id : defaults[`${scheme}Theme`] || scheme;
  }
  function markThemeChoices() {
    const choices = { mode: themeMode(), 'light-theme': schemeTheme('light'), 'dark-theme': schemeTheme('dark') };
    document.querySelectorAll('input[data-appearance]').forEach((input) => { input.checked = input.value === choices[input.name]; });
  }
  function applyTheme() {
    const mode = themeMode();
    const scheme = mode === 'system' ? (prefersDark.matches ? 'dark' : 'light') : mode;
    root.dataset.theme = schemeTheme(scheme);
    root.dataset.scheme = scheme;
    markThemeChoices();
  }
  applyTheme();
  prefersDark.addEventListener('change', () => { if (themeMode() === 'system') applyTheme(); });

  // The sun and moon button switches to the theme picked for the other scheme.
  function toggleTheme() {
    store('mode', root.dataset.scheme === 'dark' ? 'light' : 'dark');
    applyTheme();
  }

  function copy(button) {
    const label = button.querySelector('.copy-label');
    if (label) label.dataset.idle ??= label.textContent;
    const report = (copied) => {
      button.classList.add(copied ? 'st-ok' : 'st-err');
      if (label) label.textContent = copied ? label.dataset.copied : label.dataset.copyFailed;
      setTimeout(() => {
        button.classList.remove('st-ok', 'st-err');
        if (label) label.textContent = label.dataset.idle;
      }, 1500);
    };
    // The Clipboard API exists only on HTTPS and localhost.
    if (!navigator.clipboard) return report(false);
    const text = button.dataset.copyFrom ? document.querySelector(button.dataset.copyFrom).textContent : button.dataset.copy;
    navigator.clipboard.writeText(text).then(() => report(true), () => report(false));
  }

  document.addEventListener('click', (event) => {
    if (event.target.closest('[data-toggle-theme]')) toggleTheme();
    const copyButton = event.target.closest('[data-copy], [data-copy-from]');
    if (copyButton) copy(copyButton);
    const dismissButton = event.target.closest('[data-dismiss]');
    if (dismissButton) dismissButton.parentElement.remove();
  });
  let confirming = null;
  document.addEventListener('click', (event) => {
    const trigger = event.target.closest('a[href][data-confirm], button[type=submit][data-confirm]');
    const dialog = document.getElementById('confirm-dialog');
    if (!trigger || !dialog || event.button !== 0) return;
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    confirming = trigger;
    const form = dialog.querySelector('form[method=post]');
    form.action = trigger.matches('a') ? trigger.href : '';
    dialog.querySelector('#confirm-title').textContent = trigger.dataset.confirm;
    const use = dialog.querySelector('#confirm-use');
    use.textContent = trigger.dataset.confirmUse ?? '';
    use.hidden = !trigger.dataset.confirmUse;
    const detail = dialog.querySelector('#confirm-detail');
    detail.textContent = trigger.dataset.confirmDetail;
    detail.hidden = !trigger.dataset.confirmDetail;
    for (const [selector, label] of [['[data-confirm-submit]', 'confirmAction'], ['[data-confirm-dismiss]', 'confirmDismiss']]) {
      const button = dialog.querySelector(selector);
      button.dataset.idle ??= button.textContent;
      button.textContent = trigger.dataset[label] || button.dataset.idle;
      button.disabled = false;
    }
    dialog.showModal();
  });
  document.addEventListener('submit', (event) => {
    const form = event.target;
    if (!form.closest('#confirm-dialog') || form.method === 'dialog') return;
    if (event.submitter?.getAttribute('formmethod') === 'dialog') return;
    if (confirming?.matches('a')) return setTimeout(() => { event.submitter.disabled = true; });
    event.preventDefault();
    const button = confirming;
    form.closest('dialog').close();
    button.form.requestSubmit(button);
  });
  document.addEventListener('close', (event) => {
    if (event.target.id === 'confirm-dialog') confirming?.focus();
  }, true);
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) document.getElementById('confirm-dialog')?.close();
  });

  function autodismiss(alert) {
    setTimeout(() => {
      if (alert.matches(':hover, :focus-within')) autodismiss(alert);
      else alert.remove();
    }, 5000);
  }
  document.addEventListener('change', (event) => {
    if (event.target.matches('[data-autosubmit]')) event.target.form.submit();
    if (event.target.matches('input[data-appearance]')) {
      store(event.target.name, event.target.value);
      applyTheme();
    }
    // The usage chart restores the pick while it parses, so the chart does not flash.
    if (event.target.name === 'metric') {
      try { sessionStorage.setItem('metric', event.target.value); } catch { }
    }
  });
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-autodismiss]').forEach(autodismiss);
    markThemeChoices();
  });
})();
