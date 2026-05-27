import { useCallback, useEffect, useMemo, useState } from 'react';
import type { DashboardData } from './api/client';
import { apiClient } from './api/client';
import { ControlsPanel } from './components/ControlsPanel';
import { DashboardCharts } from './components/Charts';
import { DashboardSkeleton } from './components/Skeleton';
import { ErrorState } from './components/ErrorState';
import { Header } from './components/Header';
import { KpiGrid } from './components/KpiGrid';
import { LoginScreen } from './components/LoginScreen';
import { OrdersTable } from './components/OrdersTable';
import { RiskBudgetPanel } from './components/RiskBudgetPanel';
import { SignalPanel } from './components/SignalPanel';
import { Toast, type ToastState } from './components/Toast';
import { TradesTable } from './components/TradesTable';

const TOKEN_KEY = 'btc-paper-trader-admin-token';

export default function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY) ?? '');
  const [data, setData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [toast, setToast] = useState<ToastState>(null);

  const loadDashboard = useCallback(async () => {
    if (!token) return;
    setIsLoading(true);
    setError(null);
    try {
      const [healthResult, summary, market, signals, orders, trades, equityCurve] = await Promise.all([
        apiClient.health().catch(() => null),
        apiClient.summary(token),
        apiClient.market(token),
        apiClient.signals(token),
        apiClient.orders(token),
        apiClient.trades(token),
        apiClient.equityCurve(token),
      ]);
      setData({ health: healthResult, summary, market, signals, orders, trades, equityCurve });
      setLastRefreshed(new Date());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load dashboard.');
    } finally {
      setIsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  const latestSignal = useMemo(() => data?.summary.latest_signal ?? data?.signals[0] ?? null, [data]);

  function saveToken(nextToken: string) {
    sessionStorage.setItem(TOKEN_KEY, nextToken);
    setToken(nextToken);
  }

  function logout() {
    sessionStorage.removeItem(TOKEN_KEY);
    setToken('');
    setData(null);
    setError(null);
    setLastRefreshed(null);
  }

  function showSuccess(message: string) {
    setToast({ tone: 'success', message });
  }

  function showError(message: string) {
    setToast({ tone: 'error', message });
  }

  if (!token) {
    return <LoginScreen onSaveToken={saveToken} />;
  }

  return (
    <div className="min-h-screen text-zinc-100">
      <Header health={data?.health ?? null} isLoading={isLoading} lastRefreshed={lastRefreshed} onLogout={logout} onRefresh={() => void loadDashboard()} />

      <main className="mx-auto max-w-7xl space-y-5 px-4 py-6 lg:px-6">
        {isLoading && !data ? <DashboardSkeleton /> : null}

        {error && !data ? <ErrorState message={error} onRetry={() => void loadDashboard()} /> : null}

        {data ? (
          <>
            {error ? (
              <div className="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">{error}</div>
            ) : null}

            <KpiGrid summary={data.summary} />
            <RiskBudgetPanel summary={data.summary} />

            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
              <SignalPanel market={data.market} signal={latestSignal} />
              <ControlsPanel token={token} onError={showError} onRefresh={() => void loadDashboard()} onSuccess={showSuccess} />
            </div>

            <DashboardCharts equityCurve={data.equityCurve} orders={data.orders} signals={data.signals} />

            <div className="grid gap-4 xl:grid-cols-2">
              <OrdersTable orders={data.orders} />
              <TradesTable trades={data.trades} />
            </div>
          </>
        ) : null}
      </main>

      <Toast onClose={() => setToast(null)} toast={toast} />
    </div>
  );
}
