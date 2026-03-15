const map = L.map('map', {
  scrollWheelZoom: false,
  worldCopyJump: true
}).setView([44, -100], 4);

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
const sidebarEl = document.querySelector('.sidebar');
const mapEl = document.getElementById('map');

let stations = [];
let filteredStations = [];
let markers = [];

mapEl.addEventListener('mouseenter', () => map.scrollWheelZoom.enable());
mapEl.addEventListener('mouseleave', () => map.scrollWheelZoom.disable());
sidebarEl?.addEventListener('mouseenter', () => map.scrollWheelZoom.disable());

function escapeHtml(value = '') {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
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
    radius: station.noaaForecast ? 5.5 : 4.5,
    weight: station.noaaForecast ? 2 : 1,
    color: '#0c1116',
    fillColor: colorForStation(station),
    fillOpacity: 0.9
  });
  marker.bindTooltip(`${station.name}${station.noaaForecast ? ' • official NOAA forecast' : ''}`);
  marker.on('click', () => showStation(station));
  return marker;
}

function renderMarkers() {
  cluster.clearLayers();
  markers = filteredStations.map(markerFor);
  cluster.addLayers(markers);
  summaryEl.innerHTML = `Loaded <b>${filteredStations.length.toLocaleString()}</b> river sensors. Hover the map and use your mouse wheel to zoom. <span class="legend-dot" style="background:#8fd14f"></span>USGS observed <span class="legend-dot" style="background:#4dd0e1"></span>USGS + official NOAA forecast <span class="legend-dot" style="background:#ff8f3f"></span>Canada observed`;
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
  const colors = ['#4dd0e1', '#ff8f3f', '#8fd14f', '#e66cff'];
  const paths = seriesList.map((series, idx) => {
    const d = series.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${pad + (i / Math.max(1, series.points.length - 1)) * (width - pad * 2)} ${scaleY(p.y)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="2.5" />`;
  }).join('');
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${paths}</svg>`;
}

function formatDate(value) {
  if (!value) return 'n/a';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

function summarizeForecast(noaa) {
  const forecast = noaa?.forecast;
  const points = (forecast?.data || []).filter((p) => Number.isFinite(p.primary) && p.primary > -900);
  if (!forecast || !points.length) {
    return {
      quality: 'Observed only',
      issued: null,
      nextValid: null,
      crest: null,
      crestTime: null,
      series: []
    };
  }
  let crest = points[0];
  for (const point of points) {
    if (point.primary > crest.primary) crest = point;
  }
  return {
    quality: 'Official NOAA forecast',
    issued: forecast.issuedTime,
    nextValid: points[0].validTime,
    crest: crest.primary,
    crestTime: crest.validTime,
    crestFlow: crest.secondary,
    primaryUnits: forecast.primaryUnits,
    secondaryUnits: forecast.secondaryUnits,
    series: [{ label: 'Forecast stage', points: points.map((p) => ({ x: p.validTime, y: p.primary })) }]
  };
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
      const currentStageSeries = liveSeries.find((s) => /gage height/i.test(s.label));
      const currentFlowSeries = liveSeries.find((s) => /discharge/i.test(s.label));
      const currentStage = currentStageSeries?.points.at(-1)?.y;
      const currentFlow = currentFlowSeries?.points.at(-1)?.y;
      const forecast = summarizeForecast(noaa);
      body = `
        <h2>${escapeHtml(station.name)}</h2>
        <div class="meta">USGS observed • ${station.state} • ${escapeHtml(station.stationId)}${station.noaaLid ? ` • NOAA ${escapeHtml(station.noaaLid)}` : ''}</div>
        <div class="kv">
          <div class="card"><b>Current stage</b><br>${currentStage ?? 'n/a'} ${escapeHtml(currentStageSeries?.unit || '')}</div>
          <div class="card"><b>Current discharge</b><br>${currentFlow ?? 'n/a'} ${escapeHtml(currentFlowSeries?.unit || '')}</div>
          <div class="card"><b>Best forecast</b><br>${escapeHtml(forecast.quality)}</div>
          <div class="card"><b>Forecast issued</b><br>${escapeHtml(formatDate(forecast.issued))}</div>
          <div class="card"><b>Forecast crest</b><br>${forecast.crest ?? 'n/a'} ${escapeHtml(forecast.primaryUnits || '')}</div>
          <div class="card"><b>Crest time</b><br>${escapeHtml(formatDate(forecast.crestTime))}</div>
        </div>
        <p><a href="https://waterdata.usgs.gov/monitoring-location/${encodeURIComponent(station.stationId)}/" target="_blank" rel="noreferrer">USGS station page</a>${station.noaaHydrographUrl ? ` • <a href="${station.noaaHydrographUrl}" target="_blank" rel="noreferrer">NOAA hydrograph</a>` : ''}</p>
        ${renderMiniChart([
          ...[currentStageSeries, currentFlowSeries].filter(Boolean).slice(0, 2),
          ...forecast.series,
          ...dailySeries.slice(0, 1)
        ])}
        <p>Forecast priority here is: <b>official NOAA forecast</b> when matched, then USGS observed/historical context. Click anywhere on the map to snap to the nearest river sensor.</p>`;
    } else {
      const series = await fetchCanadaSeries(station.stationId);
      const latest = latestCanadaValues(series.realtime);
      const dailyPoints = (series.daily.features || []).map((f) => ({ x: f.properties.DATE, y: f.properties.LEVEL ?? f.properties.DISCHARGE })).filter((p) => p.y != null);
      body = `
        <h2>${escapeHtml(station.name)}</h2>
        <div class="meta">Environment Canada observed • ${station.state} • ${escapeHtml(station.stationId)}</div>
        <div class="kv">
          <div class="card"><b>Current level</b><br>${latest.level ?? 'n/a'}</div>
          <div class="card"><b>Current discharge</b><br>${latest.discharge ?? 'n/a'}</div>
          <div class="card"><b>Historical daily mean</b><br>last 30 values</div>
          <div class="card"><b>Forecast quality</b><br>Observed only nationally</div>
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
summaryEl.innerHTML += `<div style="margin-top:8px;color:#9fb4c7">Built ${new Date(sourceData.generatedAt).toLocaleString()} • ${sourceData.usgsStations.toLocaleString()} USGS • ${sourceData.canadaStations.toLocaleString()} Canada • ${sourceData.matchedForecastStations.toLocaleString()} official NOAA forecast matches</div>`;
