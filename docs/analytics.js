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
    title: 'Deduplicated visits (8-hour window)',
    referrer,
  };

  const script = document.createElement('script');
  script.src = 'https://gc.zgo.at/count.js';
  script.async = true;
  script.dataset.goatcounter = endpoint;
  script.dataset.analyticsStatus = 'loading';
  script.referrerPolicy = 'no-referrer';
  script.addEventListener('load', () => {
    script.dataset.analyticsStatus = 'ready';
    // Keep the standard deduplicated visit and count each visible page load separately.
    let counted = false;
    const countPageview = () => {
      if (counted || document.visibilityState !== 'visible') return;
      counted = true;
      document.removeEventListener('visibilitychange', countPageview);
      window.goatcounter.count({
        path: 'pageviews',
        title: 'Page views (PV, every load)',
        event: true,
        no_session: true,
      });
      script.dataset.analyticsStatus = 'pageview-sent';
    };
    document.addEventListener('visibilitychange', countPageview);
    countPageview();
  });
  script.addEventListener('error', () => { script.dataset.analyticsStatus = 'unavailable'; });
  document.head.appendChild(script);
})();
