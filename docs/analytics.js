(() => {
  'use strict';

  const website = document.currentScript?.dataset.websiteId;
  if (!website || location.hostname !== 'renshaojie233.github.io' ||
      !['/StreamingWAM/', '/StreamingWAM/index.html'].includes(location.pathname)) return;
  if (navigator.doNotTrack === '1' || navigator.globalPrivacyControl === true) return;

  let referrer = '';
  try {
    const source = new URL(document.referrer);
    if (source.origin !== location.origin) referrer = source.origin;
  } catch { /* Direct visits have no referrer. */ }

  // Keep one canonical page and omit query strings and detailed referral URLs.
  const payload = {
    website,
    hostname: location.hostname,
    url: '/StreamingWAM/',
    title: 'StreamingWAM project page',
    referrer,
  };
  const script = document.createElement('script');
  script.src = 'https://cloud.umami.is/script.js';
  script.async = true;
  script.dataset.websiteId = website;
  script.dataset.autoTrack = 'false';
  script.dataset.doNotTrack = 'true';
  script.dataset.analyticsStatus = 'loading';
  script.referrerPolicy = 'no-referrer';
  script.addEventListener('load', () => {
    if (typeof window.umami?.track !== 'function') {
      script.dataset.analyticsStatus = 'unavailable';
      return;
    }
    // Umami records every page load and deduplicates visitors on the server.
    let counted = false;
    const countPageview = () => {
      if (counted || document.visibilityState !== 'visible') return;
      counted = true;
      document.removeEventListener('visibilitychange', countPageview);
      window.umami.track({ ...payload });
      script.dataset.analyticsStatus = 'pageview-sent';
    };
    document.addEventListener('visibilitychange', countPageview);
    countPageview();

    document.addEventListener('click', (event) => {
      const link = event.target.closest?.('a[data-analytics-event]');
      const name = link?.dataset.analyticsEvent;
      if (!name || !/^click-[a-z-]+$/.test(name)) return;
      window.umami.track({ ...payload, name });
    }, true);
  });
  script.addEventListener('error', () => { script.dataset.analyticsStatus = 'unavailable'; });
  document.head.appendChild(script);
})();
