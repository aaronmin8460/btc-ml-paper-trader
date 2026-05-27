import { Inbox } from 'lucide-react';

export function EmptyState({ title, message }: { title: string; message?: string }) {
  return (
    <div className="flex min-h-[180px] flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-white/[0.03] p-6 text-center">
      <Inbox className="h-8 w-8 text-zinc-500" />
      <p className="mt-3 text-sm font-semibold text-zinc-200">{title}</p>
      {message ? <p className="mt-1 max-w-sm text-xs leading-5 text-zinc-500">{message}</p> : null}
    </div>
  );
}
