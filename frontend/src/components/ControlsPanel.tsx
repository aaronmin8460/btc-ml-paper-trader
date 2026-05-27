import { Bell, Bot, FlaskConical, PauseCircle, PlayCircle, Zap } from 'lucide-react';
import { useState } from 'react';
import { apiClient } from '../api/client';
import { Card, CardHeader } from './Card';
import { Button } from './Button';

type ControlAction = 'runOnce' | 'startAuto' | 'stopAuto' | 'testDiscord' | 'backtest';

type ControlsPanelProps = {
  token: string;
  onSuccess: (message: string) => void;
  onError: (message: string) => void;
  onRefresh: () => void;
};

export function ControlsPanel({ token, onSuccess, onError, onRefresh }: ControlsPanelProps) {
  const [loadingAction, setLoadingAction] = useState<ControlAction | null>(null);

  async function runAction(action: ControlAction, label: string, handler: () => Promise<unknown>, confirmMessage?: string) {
    if (confirmMessage && !window.confirm(confirmMessage)) return;
    setLoadingAction(action);
    try {
      await handler();
      onSuccess(`${label} completed.`);
      onRefresh();
    } catch (error) {
      onError(error instanceof Error ? error.message : `${label} failed.`);
    } finally {
      setLoadingAction(null);
    }
  }

  return (
    <Card>
      <CardHeader eyebrow="Controls" title="Paper Bot Actions" />
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
        <Button
          isLoading={loadingAction === 'runOnce'}
          onClick={() => runAction('runOnce', 'Run once', () => apiClient.runOnce(token), 'Run one paper-trading decision now?')}
          variant="primary"
        >
          <Zap className="h-4 w-4" />
          Run once
        </Button>
        <Button
          isLoading={loadingAction === 'startAuto'}
          onClick={() => runAction('startAuto', 'Start auto trading', () => apiClient.startAuto(token), 'Start automatic paper trading?')}
          variant="secondary"
        >
          <PlayCircle className="h-4 w-4" />
          Start auto trading
        </Button>
        <Button isLoading={loadingAction === 'stopAuto'} onClick={() => runAction('stopAuto', 'Stop auto trading', () => apiClient.stopAuto(token))} variant="danger">
          <PauseCircle className="h-4 w-4" />
          Stop auto trading
        </Button>
        <Button isLoading={loadingAction === 'testDiscord'} onClick={() => runAction('testDiscord', 'Discord test', () => apiClient.testDiscord(token))} variant="secondary">
          <Bell className="h-4 w-4" />
          Test Discord alert
        </Button>
        <Button isLoading={loadingAction === 'backtest'} onClick={() => runAction('backtest', 'Backtest', () => apiClient.backtest(token))} variant="secondary">
          <FlaskConical className="h-4 w-4" />
          Run backtest
        </Button>
      </div>
      <div className="mt-4 rounded-xl border border-btc-400/20 bg-btc-500/10 p-3 text-xs leading-5 text-btc-100/80">
        <Bot className="mr-2 inline h-4 w-4 text-btc-400" />
        Actions affect paper-trading automation only. Live trading is not available in this dashboard.
      </div>
    </Card>
  );
}
