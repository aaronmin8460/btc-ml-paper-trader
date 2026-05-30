import { AlertCircle, CheckCircle2, Clock3, PauseCircle, Radio, ShieldCheck, XCircle } from 'lucide-react';
import type { ReactNode } from 'react';
import type { TradingStatus } from '../api/client';
import { cn } from '../lib/cn';
import { formatNumber } from '../lib/formatters';
import { Badge } from './Badge';
import { Card, CardHeader } from './Card';
import { EmptyState } from './EmptyState';

type StatusTone = 'green' | 'yellow' | 'red' | 'gray';

export function TradingStatusPanel({ status }: { status: TradingStatus | null }) {
  if (!status) {
    return <EmptyState title="Trading status unavailable" message="Status appears after the dashboard API responds." />;
  }

  const tone = normalizeTone(status.state_tone);
  const countdown = cooldownCountdown(status.ioc_cooldown_expires_at);
  const title = stateLabel(status.state);

  return (
    <Card className={cn('border-l-4', toneBorderClass(tone))}>
      <CardHeader
        eyebrow="Trading Status"
        title={title}
        action={status.paper_trading_only ? <Badge tone="blue">Paper Trading Only</Badge> : <Badge tone="red">Unsafe Mode</Badge>}
      />
      <div className="grid gap-4 lg:grid-cols-[1.15fr_1fr_1fr]">
        <div className={cn('rounded-xl border bg-black/20 p-4', tonePanelClass(tone))}>
          <div className="flex items-center gap-3">
            <StatusIcon tone={tone} />
            <div>
              <p className="text-sm font-semibold text-white">{title}</p>
              <p className="mt-1 text-xs text-zinc-400">{runtimeLabel(status)}</p>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Badge tone={status.scheduler_running ? 'green' : 'red'}>{status.scheduler_running ? 'Running' : 'Stopped'}</Badge>
            <Badge tone={status.auto_trade_enabled ? 'green' : 'gray'}>{status.auto_trade_enabled ? 'Auto On' : 'Auto Off'}</Badge>
            <Badge tone={status.trading_enabled ? 'green' : 'gray'}>{status.trading_enabled ? 'Trading On' : 'Trading Off'}</Badge>
          </div>
        </div>

        <StatusBlock
          icon={<Radio className="h-4 w-4" />}
          label="Last decision"
          value={formatDecision(status.latest_decision_action, status.latest_decision_reason)}
        />

        <StatusBlock
          icon={<AlertCircle className="h-4 w-4" />}
          label="Last risk reason"
          tone={status.latest_risk_block_reason ? 'red' : 'green'}
          value={status.latest_risk_block_reason ?? 'None'}
        />

        <StatusBlock
          icon={<PauseCircle className="h-4 w-4" />}
          label="Runtime pause"
          tone={status.paused ? 'red' : 'green'}
          value={status.paused ? status.pause_reason ?? 'Paused' : 'Clear'}
          helper={status.paused_at ? `Paused ${formatTime(status.paused_at)}` : undefined}
        />

        <StatusBlock
          icon={<Clock3 className="h-4 w-4" />}
          label="IOC cooldown"
          tone={status.ioc_cooldown_active ? 'yellow' : 'green'}
          value={status.ioc_cooldown_active ? countdown : 'Inactive'}
          helper={status.ioc_cooldown_expires_at ? `Expires ${formatTime(status.ioc_cooldown_expires_at)}` : undefined}
        />

        <StatusBlock
          icon={<XCircle className="h-4 w-4" />}
          label="Recent IOC cancels"
          tone={status.ioc_cooldown_active ? 'yellow' : 'gray'}
          value={formatNumber(status.current_ioc_cancel_count, 0)}
          helper={`${formatNumber(status.ioc_cancel_lookback_seconds, 0)}s lookback`}
        />

        <StatusBlock
          icon={<ShieldCheck className="h-4 w-4" />}
          label="Model status"
          tone={modelTone(status)}
          value={modelLabel(status)}
          helper={`${status.prediction_source ?? 'unknown'} source · ${status.active_model_invalid_reason ?? 'registry ok'} · fallback trading ${status.fallback_trading_allowed ? 'allowed' : 'blocked'}`}
        />
      </div>
    </Card>
  );
}

function modelLabel(status: TradingStatus): string {
  if (status.active_model_valid) return 'Accepted';
  return status.active_model_status.replace(/_/g, ' ');
}

function modelTone(status: TradingStatus): StatusTone {
  if (status.active_model_valid) return 'green';
  if (status.active_model_status === 'stale') return status.fallback_trading_allowed ? 'yellow' : 'gray';
  return 'red';
}

function StatusBlock({
  helper,
  icon,
  label,
  tone = 'gray',
  value,
}: {
  helper?: string;
  icon: ReactNode;
  label: string;
  tone?: StatusTone;
  value: string;
}) {
  return (
    <div className={cn('rounded-xl border bg-black/20 p-4', tonePanelClass(tone))}>
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-zinc-400">
        {icon}
        {label}
      </div>
      <p className="mt-3 break-words text-base font-semibold text-white">{value}</p>
      {helper ? <p className="mt-1 break-words text-xs text-zinc-400">{helper}</p> : null}
    </div>
  );
}

function StatusIcon({ tone }: { tone: StatusTone }) {
  if (tone === 'green') return <CheckCircle2 className="h-8 w-8 text-emerald-300" />;
  if (tone === 'yellow') return <Clock3 className="h-8 w-8 text-amber-200" />;
  if (tone === 'red') return <PauseCircle className="h-8 w-8 text-rose-300" />;
  return <PauseCircle className="h-8 w-8 text-zinc-400" />;
}

function runtimeLabel(status: TradingStatus): string {
  if (status.paused) return `Automatic trading is paused${status.pause_reason ? `: ${status.pause_reason}` : '.'}`;
  if (!status.trading_enabled) return 'Order submission is disabled.';
  if (!status.auto_trade_enabled) return 'Automatic trading is paused.';
  if (status.scheduler_running === false) return 'Scheduler is stopped.';
  if (status.ioc_cooldown_active) return 'Waiting for IOC cancel cooldown.';
  if (status.latest_risk_block_reason) return 'Risk guard is blocking new orders.';
  if (status.latest_decision_action === 'hold') return 'Strategy is waiting for a stronger setup.';
  return 'Scheduler and trading controls are active.';
}

function formatDecision(action: string | null, reason: string | null): string {
  if (!action && !reason) return 'No decision yet';
  return `${action?.toUpperCase() ?? '—'} · ${reason ?? 'no reason'}`;
}

function stateLabel(state: string): string {
  const labels: Record<string, string> = {
    blocked: 'Blocked',
    cooling_down: 'Cooling Down',
    disabled: 'Disabled',
    paused: 'Paused',
    running: 'Running',
    stopped: 'Stopped',
    waiting: 'Waiting',
  };
  return labels[state] ?? state.replace(/_/g, ' ');
}

function cooldownCountdown(expiresAt: string | null): string {
  if (!expiresAt) return 'Active';
  const expires = new Date(expiresAt).getTime();
  if (Number.isNaN(expires)) return 'Active';
  const remainingSeconds = Math.max(0, Math.ceil((expires - Date.now()) / 1000));
  if (remainingSeconds <= 0) return 'Expiring now';
  const minutes = Math.floor(remainingSeconds / 60);
  const seconds = remainingSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);
}

function normalizeTone(value: string): StatusTone {
  if (value === 'green' || value === 'yellow' || value === 'red' || value === 'gray') return value;
  return 'gray';
}

function toneBorderClass(tone: StatusTone): string {
  if (tone === 'green') return 'border-l-emerald-400';
  if (tone === 'yellow') return 'border-l-amber-300';
  if (tone === 'red') return 'border-l-rose-400';
  return 'border-l-zinc-500';
}

function tonePanelClass(tone: StatusTone): string {
  if (tone === 'green') return 'border-emerald-400/20';
  if (tone === 'yellow') return 'border-amber-400/25';
  if (tone === 'red') return 'border-rose-400/25';
  return 'border-white/10';
}
