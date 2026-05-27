import { AlertTriangle } from 'lucide-react';
import { Button } from './Button';

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="glass-card mx-auto max-w-3xl rounded-3xl p-8 text-center">
      <AlertTriangle className="mx-auto h-10 w-10 text-rose-300" />
      <h2 className="mt-4 text-xl font-bold text-white">Dashboard unavailable</h2>
      <p className="mt-2 text-sm leading-6 text-zinc-400">{message}</p>
      <Button className="mt-6" onClick={onRetry} variant="primary">
        Retry
      </Button>
    </div>
  );
}
