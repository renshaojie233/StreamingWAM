(() => {
  'use strict';

  const endpoint = document.currentScript?.dataset.analyticsEndpoint;
  // Enable only after the private dashboard has been configured and verified.
  if (!endpoint || location.hostname !== 'renshaojie233.github.io' ||
      location.pathname !== '/StreamingWAM/' && location.pathname !== '/StreamingWAM/index.html') return;
  if (navigator.doNotTrack === '1' || navigator.globalPrivacyControl === true) return;

  let referrer = '';
  try {
    const source = new URL(document.referrer);
    if (source.origin !== location.origin) referrer = source.origin;
  } catch { /* Direct visits have no referrer. */ }

  window.goatcounter = {
    path: '/StreamingWAM/',
    title: 'StreamingWAM project page',
    referrer,
  };

  const script = document.createElement('script');
  script.src = 'https://gc.zgo.at/count.js';
  script.async = true;
  script.dataset.goatcounter = endpoint;
  script.referrerPolicy = 'no-referrer';
  document.head.appendChild(script);
})();
