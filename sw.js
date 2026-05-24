const CACHE = 'nylo-v1';
const PRECACHE = [
  '/trading_dashboard.html',
  '/manifest.json',
  '/nylo_icon.svg',
  '/trade_log.csv',
  'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js',
  'https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap'
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE.map(u => new Request(u, { cache: 'reload' })))).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  // Network-first for live data endpoints; cache-first for static assets
  const url = e.request.url;
  const isLive = url.includes('heartbeat.json') || url.includes('trade_log.csv') ||
                 url.includes('yahoo.com') || url.includes('coingecko') ||
                 url.includes('corsproxy') || url.includes('allorigins') ||
                 url.includes('supabase');
  if (isLive) {
    e.respondWith(
      fetch(e.request).catch(() => caches.match(e.request))
    );
    return;
  }
  e.respondWith(
    caches.match(e.request).then(cached => {
      const networkFetch = fetch(e.request).then(res => {
        if (res && res.status === 200 && e.request.method === 'GET') {
          caches.open(CACHE).then(c => c.put(e.request, res.clone()));
        }
        return res;
      });
      return cached || networkFetch;
    })
  );
});
