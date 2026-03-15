const map = L.map('map').setView([44, -100], 4);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

const cluster = L.markerClusterGroup();
map.addLayer(cluster);

const summaryEl = document.getElementById('summary');
const detailsEl = document.getElementById('details');
const searchEl = document.getElementById('search');
const forecastOnlyEl = document.getElementById('forecastOnly');

let stations = [];
let filteredStations = [];
let markers = [];

function escapeHtml(value = '') {
  return value.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function haversineKm(a, b) {
  const R = 6371;
  const dLat = (b.latitude - a.latitude) * Math.PI / 180;
  const dLon = (b.longitude - a.longitude) * Math.PI / 180;
  const lat1 = a.latitude * Math.PI / 180;
  const lat2 = b.latitude * Math.PI / 180;
  const s = Math.sin(dLat / 2) ** 2 + Math.sin(dLon / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  return 2 * R * Math.asin(Math.sqrt(s));
}

function colorForStation(station) {
  if (station.country === 'CA') return '#ff8f3f';
  if (station.noaaForecast) return '#4dd0e1';
  return '#8fd14f';
}

function markerFor(station) {
  const marker = L.circleMarker([station.latitude, station.longitude], {
    radius: 5,
    weight: 1,
    color: '#0c1116',
    fillColor: colorForStation(station),
    fillOpacity: 0.9
  });
  marker.on('click', () => showStation(station));
  return marker;
}

function renderMarkers() {
  cluster.clearLayers();
  markers = filteredStations.map(markerFor);
  cluster.addLayers(markers);
  summaryEl.innerHTML = `Loaded <b>${filteredStations.length.toLocaleString()}</b> river sensors. <span class="legend-dot" style="background:#8fd14f"></span>USGS <span class="legend-dot" style="background:#4dd0e1"></span>USGS + NOAA forecast <span class="legend-dot" style="background:#ff8f3f"></span>Canada`;
}

function applyFilters() {
  const q = searchEl.value.trim().toLowerCase();
  filteredStations = stations.filter((station) => {
    if (forecastOnlyEl.checked && !station.forecast) return false;
    if (!q) return true;
    const hay = [station.name, station.waterbody, station.state, station.stationId, station.network, station.country].join(' ').toLowerCase();
    return hay.includes(q);
  });
  renderMarkers();
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function fetchUsgsSeries(siteNo) {
  const iv = await fetchJson(`https://waterservices.usgs.gov/nwis/iv/?format=json&sites=${encodeURIComponent(siteNo)}&parameterCd=00065,00060`);
  const dv = await fetchJson(`https://waterservices.usgs.gov/nwis/dv/?format=json&sites=${encodeURIComponent(siteNo)}&parameterCd=00065,00060&period=P30D`);
  return { iv, dv };
}

async function fetchCanadaSeries(stationNo) {
  const realtime = await fetchJson(`https://api.weather.gc.ca/collections/hydrometric-realtime/items?f=json&STATION_NUMBER=${encodeURIComponent(stationNo)}&limit=288`);
  const daily = await fetchJson(`https://api.weather.gc.ca/collections/hydrometric-daily-mean/items?f=json&STATION_NUMBER=${encodeURIComponent(stationNo)}&limit=30`);
  return { realtime, daily };
}

function latestUsgsValue(json) {
  const series = json.value?.timeSeries || [];
  return series.map((s) => ({
    label: s.variable?.variableDescription,
    unit: s.variable?.unit?.unitCode,
    points: (s.values?.[0]?.value || []).map((v) => ({ x: v.dateTime, y: Number(v.value) })).filter((v) => Number.isFinite(v.y))
  }));
}

function latestCanadaValues(json) {
  const features = json.features || [];
  return {
    level: [...features].reverse().find((f) => f.properties.LEVEL != null)?.properties.LEVEL ?? null,
    discharge: [...features].reverse().find((f) => f.properties.DISCHARGE != null)?.properties.DISCHARGE ?? null,
    points: features.map((f) => ({ x: f.properties.DATETIME || f.properties.DATE, y: f.properties.LEVEL ?? f.properties.DISCHARGE })).filter((p) => p.y != null)
  };
}

function renderMiniChart(seriesList) {
  if (!seriesList.length) return '<div class="card">No chartable series available.</div>';
  const width = 640;
  const height = 220;
  const pad = 20;
  const pts = seriesList.flatMap((s) => s.points || []);
  if (!pts.length) return '<div class="card">No chartable series available.</div>';
  const ys = pts.map((p) => p.y);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scaleY = (y) => height - pad - ((y - minY) / ((maxY - minY) || 1)) * (height - pad * 2);
  const maxLen = Math.max(...seriesList.map((s) => s.points.length));
  const scaleX = (i, len) => pad + (i / Math.max(1, len - 1)) * (width - pad * 2);
  const colors = ['#4dd0e1', '#ff8f3f', '#8fd14f'];
  const paths = seriesList.map((series, idx) => {
    const d = series.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i, series.points.length)} ${scaleY(p.y)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="2" />`;
  }).join('');
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${paths}</svg>`;
}

async function showStation(station) {
  detailsEl.innerHTML = `<div class="card">Loading ${escapeHtml(station.name)}…</div>`;
  try {
    let body = '';
    if (station.country === 'US') {
      const [usgs, noaa] = await Promise.all([
        fetchUsgsSeries(station.stationId),
        station.noaaLid ? fetchJson(`data/noaa-forecast/${station.noaaLid}.json`).catch(() => null) : Promise.resolve(null)
      ]);
      const liveSeries = latestUsgsValue(usgs.iv);
      const dailySeries = latestUsgsValue(usgs.dv);
      const forecastSeries = (noaa?.forecast?.data || []).map((p) => ({ x: p.validTime, y: p.primary })).filter((p) => Number.isFinite(p.y) && p.y > -900);
      const currentStage = liveSeries.find((s) => /gage height/i.test(s.label))?.points.at(-1)?.y;
      const currentFlow = liveSeries.find((s) => /discharge/i.test(s.label))?.points.at(-1)?.y;
      body = `
        <h2>${escapeHtml(station.name)}</h2>
        <div class="meta">${station.network} • ${station.state} • ${escapeHtml(station.stationId)}${station.noaaLid ? ` • NOAA ${escapeHtml(station.noaaLid)}` : ''}</div>
        <div class="kv">
          <div class="card"><b>Current stage</b><br>${currentStage ?? 'n/a'}</div>
          <div class="card"><b>Current discharge</b><br>${currentFlow ?? 'n/a'}</div>
          <div class="card"><b>Forecast</b><br>${station.noaaLid ? 'available' : 'not matched'}</div>
          <div class="card"><b>Location</b><br>${station.latitude.toFixed(4)}, ${station.longitude.toFixed(4)}</div>
        </div>
        <p><a href="https://waterdata.usgs.gov/monitoring-location/${encodeURIComponent(station.stationId)}/" target="_blank" rel="noreferrer">USGS station page</a>${station.noaaHydrographUrl ? ` • <a href="${station.noaaHydrographUrl}" target="_blank" rel="noreferrer">NOAA hydrograph</a>` : ''}</p>
        ${renderMiniChart([...liveSeries.slice(0, 2), ...(forecastSeries.length ? [{ label: 'Forecast stage', points: forecastSeries }] : []), ...dailySeries.slice(0, 1)])}
        <p>Tip: clicking the map snaps to the nearest sensor, so you can inspect a river reach even when there is no marker exactly at your click point.</p>`;
    } else {
      const series = await fetchCanadaSeries(station.stationId);
      const latest = latestCanadaValues(series.realtime);
      const dailyPoints = (series.daily.features || []).map((f) => ({ x: f.properties.DATE, y: f.properties.LEVEL ?? f.properties.DISCHARGE })).filter((p) => p.y != null);
      body = `
        <h2>${escapeHtml(station.name)}</h2>
        <div class="meta">Environment Canada • ${station.state} • ${escapeHtml(station.stationId)}</div>
        <div class="kv">
          <div class="card"><b>Current level</b><br>${latest.level ?? 'n/a'}</div>
          <div class="card"><b>Current discharge</b><br>${latest.discharge ?? 'n/a'}</div>
          <div class="card"><b>Historical daily mean</b><br>last 30 values</div>
          <div class="card"><b>Forecast</b><br>not available from national open API</div>
        </div>
        <p><a href="https://wateroffice.ec.gc.ca/report/real_time_e.html?stn=${encodeURIComponent(station.stationId)}" target="_blank" rel="noreferrer">Environment Canada station page</a></p>
        ${renderMiniChart([{ label: 'Realtime', points: latest.points }, { label: 'Daily mean', points: dailyPoints }])}`;
    }
    detailsEl.innerHTML = body;
  } catch (error) {
    detailsEl.innerHTML = `<div class="card">Could not load sensor details. ${escapeHtml(String(error.message || error))}</div>`;
  }
}

function findNearest(latlng) {
  const point = { latitude: latlng.lat, longitude: latlng.lng };
  let nearest = null;
  let best = Infinity;
  for (const station of filteredStations.length ? filteredStations : stations) {
    const dist = haversineKm(point, station);
    if (dist < best) {
      best = dist;
      nearest = station;
    }
  }
  return nearest;
}

map.on('click', (event) => {
  const nearest = findNearest(event.latlng);
  if (nearest) {
    showStation(nearest);
    map.flyTo([nearest.latitude, nearest.longitude], Math.max(map.getZoom(), 8), { duration: 0.8 });
  }
});

searchEl.addEventListener('input', applyFilters);
forecastOnlyEl.addEventListener('change', applyFilters);

const [stationData, sourceData] = await Promise.all([
  fetchJson('data/stations.json'),
  fetchJson('data/sources.json')
]);
stations = stationData;
filteredStations = stationData;
renderMarkers();
summaryEl.innerHTML += `<div style="margin-top:8px;color:#9fb4c7">Built ${new Date(sourceData.generatedAt).toLocaleString()} • ${sourceData.usgsStations.toLocaleString()} USGS • ${sourceData.canadaStations.toLocaleString()} Canada • ${sourceData.noaaGauges.toLocaleString()} NOAA gauges indexed</div>`;
