import { Bitcoin, LogOut, RefreshCcw, ShieldCheck, Wifi, WifiOff } from 'lucide-react';
import type { BotHealth } from '../api/client';
import { formatDateTime } from '../lib/formatters';
import { Badge } from './Badge';
import { Button } from './Button';

type HeaderProps = {
  health: BotHealth | null;
  isLoading: boolean;
  lastRefreshed: Date | null;
  onRefresh: () => void;
  onLogout: () => void;
};

export function Header({ health, isLoading, lastRefreshed, onRefresh, onLogout }: HeaderProps) {
  const online = health?.status === 'ok';
  return (
    <header className="sticky top-0 z-30 border-b border-white/10 bg-ink-950/80 backdrop-blur-2xl">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 md:flex-row md:items-center md:justify-between lg:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-btc-500 text-black shadow-glow">
            <Bitcoin className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">BTC ML Paper Trader</h1>
            <p className="text-xs text-zinc-500">Private paper trading dashboard</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={online ? 'green' : 'red'}>
            {online ? <Wifi className="mr-1 h-3.5 w-3.5" /> : <WifiOff className="mr-1 h-3.5 w-3.5" />}
            {online ? 'Online' : 'Offline'}
          </Badge>
          <Badge tone="orange">
            <ShieldCheck className="mr-1 h-3.5 w-3.5" />
            Paper Trading Only
          </Badge>
          <Badge tone="blue">{health?.symbol ?? 'BTC/USD'} Only</Badge>
          <span className="text-xs text-zinc-500">Refreshed {formatDateTime(lastRefreshed?.toISOString())}</span>
          <Button aria-label="Refresh dashboard" isLoading={isLoading} onClick={onRefresh} variant="secondary">
            <RefreshCcw className="h-4 w-4" />
            Refresh
          </Button>
          <Button aria-label="Clear admin token" onClick={onLogout} variant="ghost">
            <LogOut className="h-4 w-4" />
            Logout
          </Button>
        </div>
      </div>
    </header>
  );
}
