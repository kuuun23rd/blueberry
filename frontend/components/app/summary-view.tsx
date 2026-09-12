'use client';

import { useState } from 'react';
import { jsPDF } from 'jspdf';
import { CheckIcon, DownloadIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/shadcn/utils';

export interface TranscriptEntry {
  question: string;
  answer: string;
  feedback: string | null;
}

export interface InterviewSummary {
  jobTitle: string;
  durationSeconds: number;
  questionsAsked: number;
  maxQuestions: number;
  // Full question-by-question record, in order (including follow-ups), with feedback filled
  // in by the backend once the interview wraps up. Empty if the summary screen was reached via
  // the disconnect safety net rather than the normal "interview-complete" signal (see
  // view-controller.tsx) -- the UI and PDF both just omit this section in that case.
  transcript: TranscriptEntry[];
}

interface SummaryViewProps {
  summary: InterviewSummary;
  onRestart: () => void;
}

function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

const READINESS_OPTIONS = [1, 2, 3, 4, 5];

export const SummaryView = ({
  summary,
  onRestart,
  ref,
}: React.ComponentProps<'div'> & SummaryViewProps) => {
  const [rating, setRating] = useState<number | null>(null);

  const handleDownload = () => {
    const doc = new jsPDF();
    const marginX = 20;
    const pageHeight = doc.internal.pageSize.getHeight();
    const maxWidth = doc.internal.pageSize.getWidth() - marginX * 2;
    let y = 24;

    const ensureSpace = (needed: number) => {
      if (y + needed > pageHeight - 20) {
        doc.addPage();
        y = 24;
      }
    };

    const writeWrapped = (text: string, opts: { bold?: boolean; size?: number } = {}) => {
      doc.setFont('helvetica', opts.bold ? 'bold' : 'normal');
      doc.setFontSize(opts.size ?? 11);
      const lines: string[] = doc.splitTextToSize(text, maxWidth);
      for (const line of lines) {
        ensureSpace(7);
        doc.text(line, marginX, y);
        y += 7;
      }
    };

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(16);
    doc.text('AI Interview Simulator — Practice Summary', marginX, y);
    y += 12;

    const rows: [string, string][] = [
      ['Role practiced', summary.jobTitle || 'Not specified'],
      ['Date', new Date().toLocaleString()],
      [
        'Questions answered',
        summary.maxQuestions
          ? `${summary.questionsAsked} of ${summary.maxQuestions}`
          : `${summary.questionsAsked}`,
      ],
      ['Duration', formatDuration(summary.durationSeconds)],
      ['Self-rated readiness', rating !== null ? `${rating} / 5` : 'Not rated'],
    ];
    doc.setFontSize(11);
    for (const [label, value] of rows) {
      ensureSpace(9);
      doc.setFont('helvetica', 'bold');
      doc.text(`${label}:`, marginX, y);
      doc.setFont('helvetica', 'normal');
      doc.text(value, marginX + 55, y);
      y += 9;
    }

    if (summary.transcript.length > 0) {
      y += 5;
      ensureSpace(12);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(13);
      doc.text('Question-by-question transcript', marginX, y);
      y += 10;

      summary.transcript.forEach((entry, index) => {
        ensureSpace(10);
        writeWrapped(`Q${index + 1}. ${entry.question}`, { bold: true, size: 11 });
        writeWrapped(`Your answer: ${entry.answer}`, { size: 10 });
        if (entry.feedback) {
          writeWrapped(`Feedback: ${entry.feedback}`, { size: 10 });
        }
        y += 4;
      });
    }

    const fileSafeRole = summary.jobTitle
      ? summary.jobTitle
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/(^-|-$)/g, '')
      : 'session';
    doc.save(`interview-summary-${fileSafeRole || 'session'}.pdf`);
  };

  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center">
        <div className="border-input bg-accent/40 mb-2 flex size-16 items-center justify-center rounded-full">
          <CheckIcon className="text-foreground size-8" />
        </div>

        <p className="text-foreground max-w-prose pt-1 leading-6 font-medium">
          Interview complete
        </p>
        <p className="text-muted-foreground max-w-prose pt-1 text-xs leading-5 text-pretty">
          {summary.jobTitle ? `Practice session for ${summary.jobTitle}` : 'Practice session'}
        </p>

        <div className="border-input mt-6 flex w-72 justify-between rounded-md border px-4 py-3 text-left">
          <div>
            <p className="text-muted-foreground text-xs">Questions</p>
            <p className="text-foreground text-sm font-medium">
              {summary.questionsAsked}
              {summary.maxQuestions ? ` / ${summary.maxQuestions}` : ''}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Duration</p>
            <p className="text-foreground text-sm font-medium">
              {formatDuration(summary.durationSeconds)}
            </p>
          </div>
        </div>

        <div className="mt-6 w-72">
          <p className="text-foreground text-xs font-medium">
            How ready do you feel for the real interview?
          </p>
          <div className="mt-2 flex justify-center gap-2">
            {READINESS_OPTIONS.map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setRating(value)}
                aria-label={`Rate readiness ${value} out of 5`}
                className={cn(
                  'border-input size-9 rounded-full border text-sm font-medium transition-colors',
                  rating === value
                    ? 'bg-foreground text-background'
                    : 'text-foreground hover:bg-accent'
                )}
              >
                {value}
              </button>
            ))}
          </div>
          {rating !== null && (
            <p className="text-muted-foreground mt-2 text-xs">
              Thanks — that&apos;s saved for this session.
            </p>
          )}
        </div>

        {summary.transcript.length > 0 && (
          <div className="mt-6 w-full max-w-xl px-4 text-left">
            <p className="text-foreground text-xs font-medium">Your interview, question by question</p>
            <div className="mt-2 flex max-h-96 flex-col gap-3 overflow-y-auto pr-1">
              {summary.transcript.map((entry, index) => (
                <div key={index} className="border-input rounded-md border p-3">
                  <p className="text-foreground text-sm font-medium">
                    Q{index + 1}. {entry.question}
                  </p>
                  <p className="text-muted-foreground mt-1 text-xs">
                    You answered: {entry.answer}
                  </p>
                  {entry.feedback && (
                    <p className="text-foreground mt-2 text-xs italic">
                      Feedback: {entry.feedback}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <Button
          type="button"
          size="lg"
          onClick={onRestart}
          className="mt-6 w-72 rounded-full font-mono text-xs font-bold tracking-wider uppercase"
        >
          Start another interview
        </Button>
        <Button
          type="button"
          variant="outline"
          size="lg"
          onClick={handleDownload}
          className="mt-2 w-72 rounded-full font-mono text-xs font-bold tracking-wider uppercase"
        >
          <DownloadIcon className="size-4" />
          Download as PDF
        </Button>
      </section>
    </div>
  );
};
