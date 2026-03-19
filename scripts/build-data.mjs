import { mkdir, writeFile, readFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'generated');
const cacheDir = path.join(outDir, 'cache');
const forecastDir = path.join(outDir, 'noaa-forecast');
const sensorsDir = path.join(outDir, 'sensors');

const FORCE_REFRESH = process.env.FORCE_REFRESH === '1';
const STATION_CACHE_MAX_AGE_HOURS = Number(process.env.STATION_CACHE_MAX_AGE_HOURS || 24);
const FORECAST_CACHE_MAX_AGE_HOURS = Number(process.env.FORECAST_CACHE_MAX_AGE_HOURS || 6);
const NOAA_CONCURRENCY = Number(process.env.NOAA_CONCURRENCY || 16);

const US_STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC','PR','VI','GU'];
const STATE_FIPS_TO_ABBR = {
  '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE','11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN','19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA','26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE','32':'NV','33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND','39':'OH','40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD','47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV','55':'WI','56':'WY','60':'AS','66':'GU','69':'MP','72':'PR','78':'VI'
};

async function fetchJson(url) {
  const res = await fetch(url, { headers: { 'user-agent': 'north-america-river-watch-builder' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function fetchText(url) {
  const res = await fetch(url, { headers: { 'user-agent': 'north-america-river-watch-builder' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.text();
}

async function exists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonIfExists(filePath) {
  if (!await exists(filePath)) return null;
  return JSON.parse(await readFile(filePath, 'utf8'));
}

async function writeJson(filePath, value, pretty = false) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, JSON.stringify(value, null, pretty ? 2 : 0));
}

function ageHours(isoOrMs) {
  const then = typeof isoOrMs === 'number' ? isoOrMs : new Date(isoOrMs).getTime();
  return (Date.now() - then) / 36e5;
}

async function loadCachedResource(filePath, maxAgeHours, loader, label) {
  if (!FORCE_REFRESH) {
    const cached = await readJsonIfExists(filePath);
    if (cached?.generatedAt && ageHours(cached.generatedAt) <= maxAgeHours) {
      console.log(`${label}: cache hit (${ageHours(cached.generatedAt).toFixed(1)}h old)`);
      return cached.data;
    }
  }
  const data = await loader();
  await writeJson(filePath, { generatedAt: new Date().toISOString(), data });
  console.log(`${label}: refreshed`);
  return data;
}

function parseRdb(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  const dataLines = lines.filter((line) => !line.startsWith('#'));
  if (dataLines.length < 3) return [];
  const headers = dataLines[0].split('\t');
  return dataLines.slice(2).map((line) => {
    const values = line.split('\t');
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
  });
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mapUsgsSite(row) {
  const lat = num(row.dec_lat_va || row.dec_lat_va_);
  const lon = num(row.dec_long_va || row.dec_long_va_);
  if (lat == null || lon == null) return null;
  return {
    id: `usgs:${row.site_no}`,
    key: row.site_no,
    network: 'USGS',
    country: 'US',
    stationId: row.site_no,
    name: row.station_nm,
    waterbody: row.station_nm,
    state: STATE_FIPS_TO_ABBR[row.state_cd] || row.state_cd,
    latitude: lat,
    longitude: lon,
    forecast: false,
    realtime: true,
    noaaLid: null,
    noaaForecast: false
  };
}

async function fetchUsgsStations() {
  const stations = [];
  for (const state of US_STATES) {
    const url = `https://waterservices.usgs.gov/nwis/site/?format=rdb&siteOutput=expanded&siteStatus=all&hasDataTypeCd=iv&siteType=ST&stateCd=${state}`;
    try {
      const rows = parseRdb(await fetchText(url));
      for (const row of rows) {
        const mapped = mapUsgsSite(row);
        if (mapped) stations.push(mapped);
      }
      console.log(`USGS ${state}: ${rows.length}`);
    } catch (error) {
      console.warn(`USGS ${state} failed: ${error.message}`);
    }
  }
  return stations;
}

async function fetchCanadaStations() {
  const url = 'https://api.weather.gc.ca/collections/hydrometric-stations/items?f=json&limit=10000';
  const json = await fetchJson(url);
  return (json.features || []).map((feature) => ({
    id: `ca:${feature.properties.STATION_NUMBER}`,
    key: feature.properties.STATION_NUMBER,
    network: 'ECCC',
    country: 'CA',
    stationId: feature.properties.STATION_NUMBER,
    name: feature.properties.STATION_NAME,
    waterbody: feature.properties.STATION_NAME,
    state: feature.properties.PROV_TERR_STATE_LOC,
    latitude: feature.geometry.coordinates[1],
    longitude: feature.geometry.coordinates[0],
    forecast: false,
    realtime: !!feature.properties.REAL_TIME,
    noaaLid: null,
    noaaForecast: false,
    status: feature.properties.STATUS_EN
  }));
}

async function fetchNoaaGauges() {
  const features = [];
  let offset = 0;
  const pageSize = 2000;
  while (true) {
    const url = `https://mapservices.weather.noaa.gov/eventdriven/rest/services/water/riv_gauges/MapServer/0/query?where=1%3D1&outFields=*&returnGeometry=true&f=json&resultOffset=${offset}&resultRecordCount=${pageSize}`;
    const json = await fetchJson(url);
    const page = json.features || [];
    features.push(...page);
    console.log(`NOAA gauges fetched: ${features.length}`);
    if (page.length < pageSize) break;
    offset += pageSize;
  }
  return features.map((feature) => ({
    lid: feature.attributes.gaugelid,
    usgsId: feature.attributes.usgsid || null,
    name: feature.attributes.location,
    waterbody: feature.attributes.waterbody,
    state: feature.attributes.state,
    latitude: feature.geometry?.y,
    longitude: feature.geometry?.x,
    status: feature.attributes.status,
    action: feature.attributes.action,
    flood: feature.attributes.flood,
    moderate: feature.attributes.moderate,
    major: feature.attributes.major,
    units: feature.attributes.units,
    hydrographUrl: feature.attributes.url
  })).filter((item) => item.lid && item.latitude != null && item.longitude != null);
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

function joinStations(usgsStations, canadaStations, noaaGauges) {
  const stations = [...usgsStations, ...canadaStations];
  const usgsByState = new Map();
  for (const station of usgsStations) {
    if (!usgsByState.has(station.state)) usgsByState.set(station.state, []);
    usgsByState.get(station.state).push(station);
  }

  for (const gauge of noaaGauges) {
    const candidates = usgsByState.get(gauge.state) || [];
    let best = null;
    let bestDist = Infinity;
    for (const station of candidates) {
      const dist = haversineKm(gauge, station);
      if (dist < bestDist) {
        best = station;
        bestDist = dist;
      }
    }
    if (best && bestDist <= 1.5) {
      best.noaaLid = gauge.lid;
      best.noaaForecast = true;
      best.forecast = true;
      best.noaaWaterbody = gauge.waterbody;
      best.noaaStatus = gauge.status;
      best.noaaHydrographUrl = gauge.hydrographUrl;
    }
  }
  return stations;
}

async function fetchNoaaBundle(lid) {
  const reachSeries = [
    ['analysis_assimilation', 'analysisAssimilation'],
    ['short_range', 'shortRange'],
    ['medium_range', 'mediumRange'],
    ['medium_range_blend', 'mediumRangeBlend'],
    ['long_range', 'longRange']
  ];

  const stageflow = await fetchJson(`https://api.water.noaa.gov/nwps/v1/gauges/${encodeURIComponent(lid)}/stageflow`);
  const gauge = await fetchJson(`https://api.water.noaa.gov/nwps/v1/gauges/${encodeURIComponent(lid)}`);
  const payload = {
    gauge,
    stageflow,
    modelGuidance: {},
    availableSeries: []
  };

  const officialForecastPoints = (stageflow?.forecast?.data || []).filter((p) => Number.isFinite(p?.primary) && p.primary > -900).length;
  const observedPoints = (stageflow?.observed?.data || []).filter((p) => Number.isFinite(p?.primary) && p.primary > -900).length;

  if (gauge?.reachId) {
    try {
      const reach = await fetchJson(`https://api.water.noaa.gov/nwps/v1/reaches/${encodeURIComponent(gauge.reachId)}/streamflow`);
      payload.reach = reach.reach || null;
      for (const [, key] of reachSeries) {
        const series = reach?.[key]?.series;
        const count = (series?.data || []).filter((p) => Number.isFinite(p?.flow)).length;
        if (count) {
          payload.modelGuidance[key] = series;
          payload.availableSeries.push({ type: key, points: count, units: series.units, referenceTime: series.referenceTime });
        }
      }
    } catch (error) {
      payload.reachError = error.message;
    }
  }

  if (officialForecastPoints) payload.availableSeries.unshift({ type: 'officialForecast', points: officialForecastPoints, units: stageflow?.forecast?.primaryUnits, referenceTime: stageflow?.forecast?.issuedTime });
  if (observedPoints) payload.availableSeries.unshift({ type: 'observedStage', points: observedPoints, units: stageflow?.observed?.primaryUnits, referenceTime: stageflow?.observed?.issuedTime || null });

  payload.generatedAt = new Date().toISOString();
  return payload;
}

function sensorSlug(station) {
  return station.id.replace(/[^a-zA-Z0-9_-]+/g, '-');
}

function stationIndexEntry(station) {
  return {
    id: station.id,
    slug: sensorSlug(station),
    key: station.key,
    network: station.network,
    country: station.country,
    stationId: station.stationId,
    name: station.name,
    waterbody: station.waterbody,
    state: station.state,
    latitude: station.latitude,
    longitude: station.longitude,
    forecast: station.forecast,
    realtime: station.realtime,
    noaaLid: station.noaaLid,
    noaaForecast: station.noaaForecast
  };
}

async function writeSensorFiles(stations) {
  await mkdir(sensorsDir, { recursive: true });
  const index = stations.map(stationIndexEntry);
  await writeJson(path.join(sensorsDir, 'index.json'), index);

  await Promise.all(stations.map((station) => writeJson(
    path.join(sensorsDir, `${sensorSlug(station)}.json`),
    {
      ...station,
      slug: sensorSlug(station),
      noaaForecastPath: station.noaaLid ? `data/noaa-forecast/${station.noaaLid}.json` : null,
      generatedAt: new Date().toISOString()
    },
    true
  )));
}

async function fetchNoaaForecastFiles(stations) {
  const lids = [...new Set(stations.filter((s) => s.noaaLid).map((s) => s.noaaLid))];
  await mkdir(forecastDir, { recursive: true });

  const refreshLids = [];
  let cacheHits = 0;

  for (const lid of lids) {
    const filePath = path.join(forecastDir, `${lid}.json`);
    const cached = FORCE_REFRESH ? null : await readJsonIfExists(filePath);
    if (cached?.generatedAt && ageHours(cached.generatedAt) <= FORECAST_CACHE_MAX_AGE_HOURS) {
      cacheHits += 1;
      continue;
    }
    refreshLids.push(lid);
  }

  console.log(`NOAA bundle cache hits: ${cacheHits}`);
  console.log(`NOAA bundles to refresh: ${refreshLids.length}`);

  let index = 0;
  let completed = 0;
  let failed = 0;

  async function worker() {
    while (index < refreshLids.length) {
      const lid = refreshLids[index++];
      try {
        const json = await fetchNoaaBundle(lid);
        await writeJson(path.join(forecastDir, `${lid}.json`), json);
        const summary = (json.availableSeries || []).map((s) => `${s.type}:${s.points}`).join(', ') || 'none';
        completed += 1;
        if (completed % 100 === 0 || completed === refreshLids.length) console.log(`NOAA progress ${completed}/${refreshLids.length}`);
        console.log(`NOAA forecast ${lid} (${summary})`);
      } catch (error) {
        failed += 1;
        console.warn(`NOAA forecast failed for ${lid}: ${error.message}`);
      }
    }
  }

  await Promise.all(Array.from({ length: Math.max(1, NOAA_CONCURRENCY) }, () => worker()));
  console.log(`NOAA refresh done: ${completed} ok, ${failed} failed, ${cacheHits} cached`);
}

async function main() {
  await mkdir(outDir, { recursive: true });
  await mkdir(cacheDir, { recursive: true });

  const [usgsStations, canadaStations, noaaGauges] = await Promise.all([
    loadCachedResource(path.join(cacheDir, 'usgs-stations.json'), STATION_CACHE_MAX_AGE_HOURS, fetchUsgsStations, 'USGS stations'),
    loadCachedResource(path.join(cacheDir, 'canada-stations.json'), STATION_CACHE_MAX_AGE_HOURS, fetchCanadaStations, 'Canada stations'),
    loadCachedResource(path.join(cacheDir, 'noaa-gauges.json'), STATION_CACHE_MAX_AGE_HOURS, fetchNoaaGauges, 'NOAA gauges')
  ]);

  const stations = joinStations(usgsStations, canadaStations, noaaGauges);
  await writeJson(path.join(outDir, 'stations.json'), stations);
  await writeSensorFiles(stations);

  const matchedForecastStations = stations.filter((s) => s.noaaLid).length;
  const sources = {
    generatedAt: new Date().toISOString(),
    stations: stations.length,
    usgsStations: usgsStations.length,
    canadaStations: canadaStations.length,
    noaaGauges: noaaGauges.length,
    matchedForecastStations,
    config: {
      forceRefresh: FORCE_REFRESH,
      stationCacheMaxAgeHours: STATION_CACHE_MAX_AGE_HOURS,
      forecastCacheMaxAgeHours: FORECAST_CACHE_MAX_AGE_HOURS,
      noaaConcurrency: NOAA_CONCURRENCY
    },
    notes: [
      'US current and historical series are fetched live from USGS Water Services from the browser.',
      'Canada real-time and historical daily mean series are fetched live from Environment and Climate Change Canada GeoMet APIs from the browser.',
      'NOAA NWPS forecast series are prefetched into static JSON because NOAA APIs do not expose permissive browser CORS headers.',
      'Builder is incremental: station indexes are cached and per-gauge NOAA bundles are refreshed only when stale or missing.'
    ]
  };
  await writeJson(path.join(outDir, 'sources.json'), sources, true);

  await fetchNoaaForecastFiles(stations);
  console.log(`Generated ${stations.length} stations`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
