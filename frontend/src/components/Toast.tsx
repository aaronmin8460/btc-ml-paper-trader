import { AlertTriangle, CheckCircle2, X } from 'lucide-react';
import { Button } from './Button';

export type ToastState = {
  tone: 'success' | 'error';
  message: string;
} | null;

export function Toast({ toast, onClose }: { toast: ToastState; onClose: () => void }) {
  if (!toast) return null;
  const isSuccess = toast.tone === 'success';
  const Icon = isSuccess ? CheckCircle2 : AlertTriangle;
  return (
    <div className="fixed bottom-5 right-5 z-50 max-w-sm rounded-2xl border border-white/10 bg-ink-900/95 p-4 shadow-2xl backdrop-blur">
      <div className="flex items-start gap-3">
        <Icon className={isSuccess ? 'mt-0.5 h-5 w-5 text-emerald-300' : 'mt-0.5 h-5 w-5 text-rose-300'} />
        <p className="flex-1 text-sm text-zinc-100">{toast.message}</p>
        <Button aria-label="Dismiss notification" className="h-7 w-7 rounded-lg p-0" onClick={onClose} variant="ghost">
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
