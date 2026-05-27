export function formatUsd(value: number | null | undefined, options: Intl.NumberFormatOptions = {}): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: Math.abs(value) >= 100 ? 2 : 4,
    ...options,
  }).format(value);
}

export function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${new Intl.NumberFormat('en-US', {
    maximumFractionDigits: digits,
  }).format(value * 100)}%`;
}

export function formatProbability(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${value.toFixed(4)} (${formatPercent(value, 1)})`;
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);
}

export function formatCompactTime(value: string | null | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function statusColor(actionOrStatus: string | null | undefined): string {
  const normalized = actionOrStatus?.toLowerCase() ?? '';
  if (normalized.includes('buy') || normalized.includes('filled') || normalized.includes('active')) {
    return 'text-emerald-300 bg-emerald-500/10 border-emerald-400/20';
  }
  if (normalized.includes('sell') || normalized.includes('error') || normalized.includes('rejected')) {
    return 'text-rose-300 bg-rose-500/10 border-rose-400/20';
  }
  if (normalized.includes('cancel')) {
    return 'text-amber-200 bg-amber-500/10 border-amber-400/20';
  }
  return 'text-sky-200 bg-sky-500/10 border-sky-400/20';
}

export function isPresent<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}
