// Every click is its own write. Nothing is buffered here and Next saves nothing, so the
// worst a crashed tab can cost is the message in progress.
(() => {
  const threadId = Number(document.body.dataset.threadId);
  const remainingPill = document.getElementById('remaining');

  // Time to the FIRST decision on a message, measured from when the previous one was made.
  // Noisy by nature — someone makes tea — so it is for filtering out annotators who are not
  // reading, not for scoring anyone.
  let sinceLastDecision = Date.now();

  async function send(article, label) {
    const buttons = article.querySelectorAll('button');
    buttons.forEach((b) => { b.disabled = true; });
    try {
      const res = await fetch('/api/annotate', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          thread_id: threadId,
          message_id: Number(article.dataset.messageId),
          ms_spent: Date.now() - sinceLastDecision,
          label,
        }),
      });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      sinceLastDecision = Date.now();
      article.classList.remove('assumed', 'failed');
      article.querySelectorAll('.label').forEach((b) => {
        b.classList.toggle('on', b.dataset.label === data.entry.label);
      });
      remainingPill.dataset.remaining = String(data.remaining);
      remainingPill.textContent = data.remaining === 0 ? 'complete' : `${data.remaining} left`;
    } catch {
      // Say so rather than leaving a button looking selected when nothing was written.
      article.classList.add('failed');
    } finally {
      buttons.forEach((b) => { b.disabled = false; });
    }
  }

  document.querySelectorAll('.msg').forEach((article) => {
    article.querySelector('[data-reveal]')?.addEventListener('click', () => {
      article.classList.add('open');
    });
    article.querySelectorAll('.label').forEach((button) => {
      button.addEventListener('click', () => { send(article, button.dataset.label); });
    });
  });
})();
