/* Pure aggregation logic shared by the browser and the Node verification script. */
(function (root) {
  'use strict';
  const sums = ['policies', 'exposure', 'loss', 'claims', 'fs', 'tw', 'base', 'dev_fs', 'dev_tw', 'dev_base'];
  function decode(data) { return data.cube.map(row => Object.fromEntries(data.columns.map((name, i) => [name, row[i]]))); }
  function filter(rows, selection) { return rows.filter(row => ['age', 'region', 'vehicle'].every(key => selection[key] === 'all' || row[key] === Number(selection[key]))); }
  function aggregate(rows) {
    const result = Object.fromEntries(sums.map(key => [key, 0]));
    for (const row of rows) for (const key of sums) result[key] += row[key];
    return result;
  }
  function group(rows, key, count, start = 0) { return Array.from({length: count}, (_, i) => ({group: i + start, ...aggregate(rows.filter(row => row[key] === i + start))})); }
  function ratio(numerator, denominator) { return denominator > 0 ? numerator / denominator : null; }
  const api = {decode, filter, aggregate, group, ratio};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LabMath = api;
})(typeof window !== 'undefined' ? window : globalThis);
