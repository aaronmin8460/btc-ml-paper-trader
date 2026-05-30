export type BotHealth = {
  status: string;
  paper_trading_only: boolean;
  symbol: string;
};

export type SignalAction = 'buy' | 'sell' | 'hold' | string;

export type DashboardSignal = {
  created_at: string | null;
  symbol: string;
  action: SignalAction;
  buy_probability: number | null;
  sell_probability: number | null;
  reason: string | null;
};

export type DashboardOrder = {
  created_at: string | null;
  symbol: string;
  side: 'buy' | 'sell' | string;
  status: string | null;
  notional: number | null;
  qty: number | null;
  broker_order_id: string | null;
  raw_response?: unknown;
};

export type DashboardTrade = {
  created_at: string | null;
  symbol: string;
  side: 'buy' | 'sell' | string;
  qty: number | null;
  price: number | null;
  pnl: number | null;
};

export type EquityPoint = {
  timestamp: string | null;
  trade_pnl: number | null;
  cumulative_realized_pnl: number | null;
  drawdown: number | null;
};

export type AccountSnapshot = {
  created_at: string | null;
  equity: number | null;
  cash: number | null;
  buying_power: number | null;
  portfolio_value: number | null;
  currency: string | null;
  raw_response?: unknown;
};

export type PortfolioPoint = {
  timestamp: string | null;
  equity: number | null;
  cash: number | null;
  buying_power: number | null;
  portfolio_value: number | null;
};

export type TradingStatus = {
  state: 'running' | 'waiting' | 'cooling_down' | 'blocked' | 'paused' | 'stopped' | 'disabled' | string;
  state_tone: 'green' | 'yellow' | 'red' | 'gray' | string;
  paused: boolean;
  pause_reason: string | null;
  paused_at: string | null;
  latest_decision_action: string | null;
  latest_decision_reason: string | null;
  latest_risk_block_reason: string | null;
  current_ioc_cancel_count: number | null;
  ioc_cancel_lookback_seconds: number | null;
  ioc_cooldown_active: boolean;
  ioc_cooldown_expires_at: string | null;
  scheduler_running: boolean | null;
  auto_trade_enabled: boolean;
  trading_enabled: boolean;
  paper_trading_only: boolean;
  model_available: boolean;
  prediction_source: string | null;
  active_model_path: string | null;
  active_model_version: string | null;
  active_model_status: string;
  active_model_valid: boolean;
  active_model_invalid_reason: string | null;
  active_model_promotion_reason: string | null;
  active_model_net_return_pct: number | null;
  active_model_profit_factor_net: number | null;
  active_model_number_of_trades: number | null;
  registry_metadata_matches_joblib: boolean;
  active_model_registry_mismatched: boolean;
  fallback_trading_allowed: boolean;
};

export type PositionSummary = {
  symbol: string;
  qty: number | null;
  avg_entry_price: number | null;
  market_value: number | null;
  opened_at: string | null;
  highest_price: number | null;
  realized_pnl_today: number | null;
  drawdown_pct: number | null;
  last_loss_at: string | null;
};

export type AccountSummary = {
  status: string | null;
  currency: string | null;
  buying_power: number | null;
  cash: number | null;
  equity: number | null;
  portfolio_value: number | null;
  last_equity?: number | null;
  daily_change_usd?: number | null;
  daily_change_pct?: number | null;
  drawdown_pct?: number | null;
  paper: boolean | null;
} | null;

export type DataFreshness = {
  latest_timestamp: string | null;
  current_utc_time: string | null;
  latest_bar_age_seconds: number | null;
  cache_age_seconds: number | null;
};

export type DashboardSummary = {
  app_status: string;
  symbol: string;
  paper_trading_only: boolean;
  trading_enabled: boolean;
  auto_trade_enabled: boolean;
  scheduler_running: boolean | null;
  latest_btc_price: number | null;
  latest_signal: DashboardSignal | null;
  current_position: PositionSummary;
  profit_guard_enabled: boolean;
  min_net_exit_profit_pct: number;
  current_unrealized_pnl_pct: number | null;
  profit_guard_exit_allowed: boolean;
  estimated_exit_price: number | null;
  minimum_profitable_exit_price: number | null;
  alpaca_account: AccountSummary;
  alpaca_calls_last_minute: number | null;
  alpaca_budget_remaining: number | null;
  alpaca_endpoint_counts: Record<string, number> | null;
  api_budget_status: string | null;
  account_equity: number | null;
  cash: number | null;
  buying_power: number | null;
  portfolio_value: number | null;
  account_daily_change_usd: number | null;
  account_daily_change_pct: number | null;
  account_drawdown_pct: number | null;
  latest_model_net_return_pct: number | null;
  latest_model_max_drawdown_pct: number | null;
  latest_model_profit_factor: number | null;
  latest_model_accepted: boolean | null;
  latest_model_rejected_reason: string | null;
  active_model_path: string | null;
  active_model_version: string | null;
  active_model_status: string;
  active_model_valid: boolean;
  active_model_invalid_reason: string | null;
  active_model_promotion_reason: string | null;
  active_model_net_return_pct: number | null;
  active_model_profit_factor_net: number | null;
  active_model_number_of_trades: number | null;
  registry_metadata_matches_joblib: boolean;
  active_model_registry_mismatched: boolean;
  total_orders: number;
  total_buy_orders: number;
  total_sell_orders: number;
  total_trades: number;
  closed_trades: number;
  total_realized_pnl: number | null;
  total_return_pct: number | null;
  unrealized_pnl: number | null;
  win_rate: number | null;
  average_trade_pnl: number | null;
  best_trade_pnl: number | null;
  worst_trade_pnl: number | null;
  max_drawdown: number | null;
  last_order: DashboardOrder | null;
  last_trade: DashboardTrade | null;
  data_freshness: DataFreshness;
};

export type DashboardMarket = {
  symbol: string;
  timeframe: string;
  latest_close: number | null;
  latest_timestamp: string | null;
  current_utc_time: string | null;
  latest_quote: Record<string, unknown>;
  bid_price: number | null;
  ask_price: number | null;
  mid_price: number | null;
  spread_bps: number | null;
  quote_imbalance: number | null;
  cache_age_seconds: number | null;
  latest_bar_age_seconds: number | null;
  bars_count: number;
};

export type DashboardRunOnceResult = {
  prediction?: {
    buy_probability?: number | null;
    sell_probability?: number | null;
    features?: Record<string, unknown>;
  };
  decision?: {
    action?: string | null;
    reason?: string | null;
  };
  order?: {
    status?: string | null;
  } | null;
  summary: {
    action: string | null;
    reason: string | null;
    buy_probability: number | null;
    sell_probability: number | null;
    order_status: string | null;
    latest_price: number | null;
  };
};

export type DashboardData = {
  health: BotHealth | null;
  summary: DashboardSummary;
  market: DashboardMarket;
  signals: DashboardSignal[];
  orders: DashboardOrder[];
  trades: DashboardTrade[];
  equityCurve: EquityPoint[];
  accountSnapshots: AccountSnapshot[];
  portfolioCurve: PortfolioPoint[];
  tradingStatus: TradingStatus;
};

export type ActionResult = {
  ok?: boolean;
  [key: string]: unknown;
};

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') ?? '';

function apiUrl(path: string): string {
  return `${configuredBaseUrl}${path}`;
}

async function fetchJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...options,
    headers: {
      Accept: 'application/json',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    const detail = await safeErrorText(response);
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }

  return (await response.json()) as T;
}

function authHeaders(adminToken: string): HeadersInit {
  if (!adminToken.trim()) {
    throw new Error('Admin token is required.');
  }
  return { 'X-Admin-Token': adminToken };
}

async function protectedJson<T>(path: string, adminToken: string, options: RequestInit = {}): Promise<T> {
  return fetchJson<T>(path, {
    ...options,
    headers: {
      ...authHeaders(adminToken),
      ...(options.headers ?? {}),
    },
  });
}

async function safeErrorText(response: Response): Promise<string> {
  try {
    const contentType = response.headers.get('content-type') ?? '';
    if (contentType.includes('application/json')) {
      const body = (await response.json()) as { detail?: unknown };
      return typeof body.detail === 'string' ? body.detail : JSON.stringify(body);
    }
    return await response.text();
  } catch {
    return '';
  }
}

export const apiClient = {
  health: () => fetchJson<BotHealth>('/health'),
  summary: (token: string) => protectedJson<DashboardSummary>('/dashboard/summary', token),
  market: (token: string) => protectedJson<DashboardMarket>('/dashboard/market', token),
  signals: (token: string, limit = 200) => protectedJson<DashboardSignal[]>(`/dashboard/signals?limit=${limit}`, token),
  orders: (token: string, limit = 200) => protectedJson<DashboardOrder[]>(`/dashboard/orders?limit=${limit}`, token),
  trades: (token: string, limit = 200) => protectedJson<DashboardTrade[]>(`/dashboard/trades?limit=${limit}`, token),
  equityCurve: (token: string) => protectedJson<EquityPoint[]>('/dashboard/equity-curve', token),
  accountSnapshots: (token: string, limit = 500) => protectedJson<AccountSnapshot[]>(`/dashboard/account-snapshots?limit=${limit}`, token),
  portfolioCurve: (token: string) => protectedJson<PortfolioPoint[]>('/dashboard/portfolio-curve', token),
  tradingStatus: (token: string) => protectedJson<TradingStatus>('/dashboard/trading-status', token),
  runOnce: (token: string) => protectedJson<DashboardRunOnceResult>('/dashboard/run-once', token, { method: 'POST' }),
  startAuto: (token: string) => protectedJson<ActionResult>('/auto/start', token, { method: 'POST' }),
  stopAuto: (token: string) => protectedJson<ActionResult>('/auto/stop', token, { method: 'POST' }),
  testDiscord: (token: string) => protectedJson<ActionResult>('/alerts/discord/test', token, { method: 'POST' }),
  backtest: (token: string) => protectedJson<ActionResult>('/backtest', token, { method: 'POST' }),
};
