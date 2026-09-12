'use client';

import { LoaderIcon } from 'lucide-react';

export const PreparingView = ({ ref }: React.ComponentProps<'div'>) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center">
        <LoaderIcon className="text-foreground mb-4 size-10 animate-spin" />
        <p className="text-foreground max-w-prose pt-1 leading-6 font-medium">
          Connecting you with your interviewer…
        </p>
        <p className="text-muted-foreground max-w-prose pt-1 text-xs leading-5 text-pretty">
          This only takes a moment. Make sure to allow microphone access if your browser asks.
        </p>
      </section>
    </div>
  );
};
