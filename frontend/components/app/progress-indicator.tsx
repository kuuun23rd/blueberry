'use client';

interface ProgressIndicatorProps {
  questionsAsked: number;
  maxQuestions: number;
}

export function ProgressIndicator({ questionsAsked, maxQuestions }: ProgressIndicatorProps) {
  if (maxQuestions <= 0) return null;

  const fraction = Math.min(questionsAsked / maxQuestions, 1);

  return (
    <div className="pointer-events-none fixed top-4 left-1/2 z-50 w-48 -translate-x-1/2">
      <div className="bg-background/80 border-input rounded-full border px-3 py-1.5 shadow-sm backdrop-blur">
        <p className="text-muted-foreground text-center text-[11px] font-medium">
          Question {Math.min(questionsAsked + 1, maxQuestions)} of {maxQuestions}
        </p>
        <div className="bg-accent mt-1 h-1 w-full overflow-hidden rounded-full">
          <div
            className="bg-foreground h-full rounded-full transition-all duration-300"
            style={{ width: `${fraction * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
