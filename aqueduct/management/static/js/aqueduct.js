(() => {
  const root = document.documentElement;
  const storageKey = 'selectedTheme';

  let stored = null;
  try { stored = localStorage.getItem(storageKey); } catch { }
  root.dataset.theme = stored === 'dark' ? 'dark' : 'light';

  function toggleTheme() {
    const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = theme;
    try { localStorage.setItem(storageKey, theme); } catch { }
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
    navigator.clipboard.writeText(button.dataset.copy).then(() => report(true), () => report(false));
  }

  document.addEventListener('click', (event) => {
    if (event.target.closest('[data-toggle-theme]')) toggleTheme();
    const copyButton = event.target.closest('[data-copy]');
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
  });
  // The icon rail (1024 to 1279px) names each icon in a tooltip beside it; Escape dismisses it.
  const rail = matchMedia('(64rem <= width < 80rem)');
  let railTip = null;
  function hideRailTip() {
    railTip?.remove();
    railTip = null;
  }
  function showRailTip(host) {
    hideRailTip();
    if (!rail.matches) return;
    const box = host.getBoundingClientRect();
    railTip = Object.assign(document.createElement('div'), { className: 'aq-rail-tip', textContent: host.dataset.tip });
    railTip.setAttribute('aria-hidden', 'true');
    railTip.style.top = `${box.top + box.height / 2}px`;
    railTip.style.left = `${box.right + 8}px`;
    document.body.append(railTip);
  }
  for (const [enter, leave] of [['pointerover', 'pointerout'], ['focusin', 'focusout']]) {
    document.addEventListener(enter, (event) => {
      const host = event.target.closest?.('.aq-sidebar [data-tip]');
      if (host) showRailTip(host);
    });
    document.addEventListener(leave, (event) => {
      if (event.target.closest?.('.aq-sidebar [data-tip]')) hideRailTip();
    });
  }
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') hideRailTip();
  });
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-autodismiss]').forEach(autodismiss);
  });
})();
