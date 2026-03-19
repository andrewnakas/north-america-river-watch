const DATA_BASES = [
  'data',
  'https://cdn.jsdelivr.net/gh/andrewnakas/north-america-river-watch@sensor-data/data'
];

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
let detailMap = null;
let activeStationId = null;

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

async function fetchJsonWithFallback(relativePath) {
  let lastError = null;
  for (const base of DATA_BASES) {
    const url = `${base.replace(/\/$/, '')}/${relativePath.replace(/^\//, '')}`;
    try {
      return await fetchJson(url);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error(`Unable to load ${relativePath}`);
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

function formatAxisDate(value, includeHour = false) {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return 'n/a';
  return d.toLocaleDateString(undefined, includeHour
    ? { month: 'short', day: 'numeric', hour: 'numeric' }
    : { month: 'short', day: 'numeric' });
}

function renderMiniChart(seriesList, title = 'River chart', subtitle = '', options = {}) {
  if (!seriesList.length) return '<div class="card">No chartable series available.</div>';

  const palette = {
    observed: '#4dd0e1',
    forecast: '#ffb14a',
    history: '#8fd14f',
    secondary: '#e66cff',
    now: '#ffd84d'
  };

  const activeSeries = seriesList.map((series, idx) => ({
    ...series,
    style: series.style || (idx === 0 ? 'observed' : idx === 1 ? 'forecast' : idx === 2 ? 'history' : 'secondary'),
    points: (series?.points || []).map((point, index) => ({
      ...point,
      xMs: Number.isFinite(new Date(point.x).getTime()) ? new Date(point.x).getTime() : index
    })).filter((point) => Number.isFinite(point.y))
  })).filter((s) => s.points.length);

  if (!activeSeries.length) return '<div class="card">No chartable series available.</div>';

  const nowMs = Date.now();
  const width = 900;
  const height = options.compact ? 300 : 360;
  const left = 54;
  const right = 18;
  const top = 18;
  const bottom = 52;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const pts = activeSeries.flatMap((s) => s.points || []);
  const ys = pts.map((p) => p.y).filter((v) => Number.isFinite(v));
  const xs = pts.map((p) => p.xMs).filter((v) => Number.isFinite(v));
  if (!ys.length || !xs.length) return '<div class="card">No chartable series available.</div>';

  const rawMinX = Math.min(...xs);
  const rawMaxX = Math.max(...xs);
  const rangeX = Math.max(rawMaxX - rawMinX, 1);
  const minX = Math.min(rawMinX, nowMs);
  const maxX = Math.max(rawMaxX, nowMs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const yPad = Math.max((maxY - minY) * 0.1, 0.5);
  const lowY = minY - yPad;
  const highY = maxY + yPad;

  const scaleY = (y) => top + plotHeight - ((y - lowY) / ((highY - lowY) || 1)) * plotHeight;
  const scaleX = (x) => left + ((x - minX) / ((maxX - minX) || 1)) * plotWidth;

  const yTicks = Array.from({ length: 5 }, (_, i) => lowY + (i / 4) * (highY - lowY));
  const yGrid = yTicks.map((tick) => {
    const y = scaleY(tick);
    return `<line class="grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" /><text class="axis-label" x="${left - 8}" y="${y + 4}" text-anchor="end">${tick.toFixed(1)}</text>`;
  }).join('');

  const xTickCount = options.compact ? 4 : 5;
  const xTicks = Array.from({ length: xTickCount }, (_, i) => {
    const frac = xTickCount === 1 ? 0 : i / (xTickCount - 1);
    const xValue = minX + frac * (maxX - minX);
    return {
      x: scaleX(xValue),
      label: formatAxisDate(xValue, rangeX < 3 * 86400000)
    };
  }).map((tick) => `<text class="axis-label" x="${tick.x}" y="${height - 12}" text-anchor="middle">${escapeHtml(tick.label)}</text>`).join('');

  const paths = activeSeries.map((series) => {
    const d = series.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.xMs)} ${scaleY(p.y)}`).join(' ');
    const stroke = palette[series.style] || palette.secondary;
    const dash = series.style === 'forecast' ? '7 5' : '';
    return `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ''} />`;
  }).join('');

  const endPoints = activeSeries.map((series) => {
    const point = series.points.at(-1);
    if (!point) return '';
    const stroke = palette[series.style] || palette.secondary;
    return `<circle cx="${scaleX(point.xMs)}" cy="${scaleY(point.y)}" r="4.5" fill="${stroke}" stroke="#08111b" stroke-width="2" />`;
  }).join('');

  const nowInsideRange = nowMs >= minX && nowMs <= maxX;
  const nowLine = nowInsideRange ? `<line class="now-line" x1="${scaleX(nowMs)}" y1="${top}" x2="${scaleX(nowMs)}" y2="${top + plotHeight}" /><text class="now-label" x="${Math.min(scaleX(nowMs) + 6, width - 52)}" y="${top + 14}">Now</text>` : '';

  const legend = activeSeries.map((series) => {
    const swatch = palette[series.style] || palette.secondary;
    const klass = series.style === 'forecast' ? 'chart-legend-line chart-legend-line-forecast' : 'chart-legend-line';
    return `<span class="chart-legend-item"><span class="${klass}" style="--legend-color:${swatch}"></span>${escapeHtml(series.label)}</span>`;
  }).join('');

  const nowLegend = nowInsideRange ? `<span class="chart-legend-item"><span class="chart-legend-line chart-legend-line-now" style="--legend-color:${palette.now}"></span>Now</span>` : '';

  return `
    <div class="chart-wrap ${options.compact ? 'chart-wrap-compact' : ''}">
      <div class="chart-head">
        <div>
          <div class="chart-title">${escapeHtml(title)}</div>
          ${subtitle ? `<div class="chart-subtitle">${escapeHtml(subtitle)}</div>` : ''}
        </div>
      </div>
      <svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="${escapeHtml(title)}">
        <line class="axis-line" x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" />
        <line class="axis-line" x1="${left}" y1="${top + plotHeight}" x2="${width - right}" y2="${top + plotHeight}" />
        ${yGrid}
        ${nowLine}
        ${paths}
        ${endPoints}
        ${xTicks}
        <text class="axis-label axis-label-bottom" x="${left}" y="${height - 30}">Time</text>
      </svg>
      <div class="chart-legend">${legend}${nowLegend}</div>
    </div>`;
}

function formatDate(value) {
  if (!value) return 'n/a';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

function renderSiteMapCard(station) {
  return `
    <div class="site-map-card card">
      <div class="site-map-head">
        <div>
          <b>Site map</b><br>
          <span class="site-map-subtitle">Station location and nearby context</span>
        </div>
        <div class="site-map-coords">${station.latitude.toFixed(4)}, ${station.longitude.toFixed(4)}</div>
      </div>
      <div id="detailSiteMap" class="detail-site-map" aria-label="Station site map"></div>
    </div>`;
}

function mountDetailMap(station) {
  const target = document.getElementById('detailSiteMap');
  if (!target) return;
  if (detailMap) {
    detailMap.remove();
    detailMap = null;
  }
  detailMap = L.map(target, {
    zoomControl: false,
    attributionControl: false,
    dragging: true,
    scrollWheelZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    tap: false
  }).setView([station.latitude, station.longitude], 11);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(detailMap);

  L.circleMarker([station.latitude, station.longitude], {
    radius: 8,
    weight: 2,
    color: '#08111b',
    fillColor: colorForStation(station),
    fillOpacity: 0.95
  }).addTo(detailMap);
}

function summarizeForecast(noaa) {
  const stageflow = noaa?.stageflow || noaa || {};
  const observed = stageflow?.observed;
  const forecast = stageflow?.forecast;
  const modelGuidance = noaa?.modelGuidance || {};

  const observedPoints = (observed?.data || []).filter((p) => Number.isFinite(p.primary) && p.primary > -900);
  const officialPoints = (forecast?.data || []).filter((p) => Number.isFinite(p.primary) && p.primary > -900);
  const observedSeries = observedPoints.length
    ? [{ label: 'Observed stage', style: 'observed', points: observedPoints.map((p) => ({ x: p.validTime, y: p.primary })) }]
    : [];

  const officialForecastSeries = officialPoints.length
    ? [{ label: 'Official NOAA forecast', style: 'forecast', points: officialPoints.map((p) => ({ x: p.validTime, y: p.primary })) }]
    : [];

  const nwmSeriesDefs = [
    ['analysisAssimilation', 'NWM analysis', 'history'],
    ['shortRange', 'NWM short-range', 'forecast'],
    ['mediumRange', 'NWM medium-range', 'secondary'],
    ['mediumRangeBlend', 'NWM medium blend', 'secondary'],
    ['longRange', 'NWM long-range', 'secondary']
  ];

  const modelSeries = nwmSeriesDefs.map(([key, label, style]) => {
    const series = modelGuidance?.[key];
    const points = (series?.data || []).filter((p) => Number.isFinite(p?.flow)).map((p) => ({ x: p.validTime, y: p.flow }));
    if (!points.length) return null;
    return {
      label,
      style,
      units: series.units,
      referenceTime: series.referenceTime,
      points
    };
  }).filter(Boolean);

  const primaryForecastSeries = officialForecastSeries.length
    ? officialForecastSeries
    : modelSeries.filter((series) => /short-range|medium-range|medium blend|long-range/i.test(series.label));

  const mainSeries = [
    ...observedSeries,
    ...primaryForecastSeries.slice(0, 3)
  ];

  const allSeries = [
    ...observedSeries,
    ...modelSeries,
    ...officialForecastSeries
  ];

  const crestSource = (officialForecastSeries[0]?.points || primaryForecastSeries[0]?.points || []).slice();
  let crest = null;
  for (const point of crestSource) {
    if (!crest || point.y > crest.y) crest = point;
  }

  const firstForecastPoint = (officialForecastSeries[0]?.points || primaryForecastSeries[0]?.points || [])[0] || null;
  const lastForecastPoint = (officialForecastSeries[0]?.points || primaryForecastSeries[0]?.points || []).at?.(-1) || null;
  const horizonDays = firstForecastPoint && lastForecastPoint
    ? Math.max(0, Math.round((new Date(lastForecastPoint.x) - new Date(firstForecastPoint.x)) / 86400000))
    : 0;

  let quality = 'Observed only';
  if (officialForecastSeries.length) quality = 'Official NOAA forecast';
  else if (primaryForecastSeries.length) quality = 'National Water Model guidance';
  else if (observedSeries.length) quality = 'Observed only';

  return {
    quality,
    issued: forecast?.issuedTime || modelSeries[1]?.referenceTime || modelSeries[0]?.referenceTime || null,
    nextValid: firstForecastPoint?.x || null,
    crest: crest?.y ?? null,
    crestTime: crest?.x ?? null,
    crestFlow: null,
    primaryUnits: forecast?.primaryUnits || primaryForecastSeries[0]?.units || observed?.primaryUnits || '',
    secondaryUnits: forecast?.secondaryUnits || '',
    horizonDays,
    observedSeries,
    forecastSeries: primaryForecastSeries,
    officialForecastSeries,
    modelSeries,
    series: allSeries,
    mainSeries
  };
}

async function loadStation(stationOrId) {
  if (typeof stationOrId === 'object' && stationOrId?.stationId) return stationOrId;
  const station = stations.find((item) => item.id === stationOrId || item.slug === stationOrId);
  if (!station) throw new Error('Unknown station');
  return fetchJsonWithFallback(`sensors/${encodeURIComponent(station.slug)}.json`);
}

async function showStation(stationOrId) {
  const baseStation = typeof stationOrId === 'object' ? stationOrId : stations.find((item) => item.id === stationOrId || item.slug === stationOrId);
  const loadingName = baseStation?.name || 'sensor';
  detailsEl.innerHTML = `<div class="card">Loading ${escapeHtml(loadingName)}…</div>`;
  try {
    const station = await loadStation(stationOrId);
    activeStationId = station.id;
    let body = '';
    if (station.country === 'US') {
      const [usgs, noaa] = await Promise.all([
        fetchUsgsSeries(station.stationId),
        station.noaaForecastPath ? fetchJsonWithFallback(station.noaaForecastPath.replace(/^data\//, '')).catch(() => null) : Promise.resolve(null)
      ]);
      const liveSeries = latestUsgsValue(usgs.iv);
      const dailySeries = latestUsgsValue(usgs.dv);
      const currentStageSeries = liveSeries.find((s) => /gage height/i.test(s.label));
      const currentFlowSeries = liveSeries.find((s) => /discharge/i.test(s.label));
      const currentStage = currentStageSeries?.points.at(-1)?.y;
      const currentFlow = currentFlowSeries?.points.at(-1)?.y;
      const forecast = summarizeForecast(noaa);
      const recentDailyStage = dailySeries.find((s) => /gage height/i.test(s.label));
      const recentDailyFlow = dailySeries.find((s) => /discharge/i.test(s.label));
      const usingModelGuidance = forecast.quality === 'National Water Model guidance';
      const primaryObservedSeries = usingModelGuidance
        ? (currentFlowSeries ? { ...currentFlowSeries, label: 'USGS current discharge', style: 'observed' } : null)
        : (currentStageSeries ? { ...currentStageSeries, label: 'USGS current stage', style: 'observed' } : null);
      const primaryHistorySeries = usingModelGuidance
        ? (recentDailyFlow ? { ...recentDailyFlow, label: 'Recent daily discharge', style: 'history' } : null)
        : (recentDailyStage ? { ...recentDailyStage, label: 'Recent daily stage', style: 'history' } : null);
      const detailChartSeries = usingModelGuidance
        ? [primaryObservedSeries, ...forecast.forecastSeries.slice(0, 3)].filter(Boolean)
        : (forecast.mainSeries.length ? forecast.mainSeries : forecast.series);
      const overviewChartSeries = [
        primaryObservedSeries,
        ...forecast.forecastSeries.slice(0, 3),
        primaryHistorySeries
      ].filter(Boolean);
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
          <div class="card"><b>Forecast horizon</b><br>${forecast.horizonDays ? `~${forecast.horizonDays} days` : 'n/a'}</div>
        </div>
        <div class="detail-grid">
          ${renderSiteMapCard(station)}
          ${renderMiniChart(detailChartSeries, usingModelGuidance ? 'River flow guidance' : 'River forecast hydrograph', forecast.quality === 'Official NOAA forecast'
            ? (forecast.horizonDays ? `Official NOAA forecast with observed context and a clear now marker. Forecast extends roughly ${forecast.horizonDays} days.` : 'Official NOAA forecast with observed context and a clear now marker.')
            : forecast.quality === 'National Water Model guidance'
              ? 'No official NOAA stage forecast was available here, so this uses National Water Model flow guidance with discharge context.'
              : 'Showing observed river conditions with a clear now marker.', { compact: true })}
        </div>
        <p><a href="https://waterdata.usgs.gov/monitoring-location/${encodeURIComponent(station.stationId)}/" target="_blank" rel="noreferrer">USGS station page</a>${station.noaaHydrographUrl ? ` • <a href="${station.noaaHydrographUrl}" target="_blank" rel="noreferrer">NOAA hydrograph</a>` : ''}</p>
        ${renderMiniChart(overviewChartSeries, usingModelGuidance ? 'Easy-read flow guidance' : 'Easy-read hydrograph', forecast.quality === 'Official NOAA forecast'
          ? 'Mobile-friendly view of stage history, official forecast, and a vertical now marker.'
          : forecast.quality === 'National Water Model guidance'
            ? 'Mobile-friendly view of recent discharge plus model guidance when official stage forecasts are unavailable.'
            : 'Mobile-friendly view of recent stage and a vertical now marker.')}
        <p>Forecast priority here is: <b>official NOAA forecast</b> first, then <b>National Water Model guidance</b> when NOAA stage forecasts are empty, then observed/historical context. Click anywhere on the map to snap to the nearest river sensor.</p>`;
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
        ${renderMiniChart([
          { label: 'Realtime level', style: 'observed', points: latest.points },
          { label: 'Daily mean', style: 'history', points: dailyPoints }
        ], 'Observed hydrograph', 'Environment Canada realtime plus recent daily mean history with a clear now marker.')}`;
    }
    detailsEl.innerHTML = body;
    if (station.country === 'US') mountDetailMap(station);
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
  fetchJsonWithFallback('sensors/index.json'),
  fetchJsonWithFallback('sources.json')
]);
stations = stationData;
filteredStations = stationData;
renderMarkers();
summaryEl.innerHTML += `<div style="margin-top:8px;color:#9fb4c7">Built ${new Date(sourceData.generatedAt).toLocaleString()} • ${sourceData.usgsStations.toLocaleString()} USGS • ${sourceData.canadaStations.toLocaleString()} Canada • ${sourceData.matchedForecastStations.toLocaleString()} official NOAA forecast matches • sensor details load on demand</div>`;
