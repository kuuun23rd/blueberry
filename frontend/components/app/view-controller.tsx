'use client';

import { useEffect, useRef, useState } from 'react';
import { useTheme } from 'next-themes';
import { AnimatePresence, motion } from 'motion/react';
import { toast } from 'sonner';
import {
  useChat,
  useDataChannel,
  useRoomContext,
  useSessionContext,
} from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AgentSessionView_01 } from '@/components/agents-ui/blocks/agent-session-view-01';
import { PreparingView } from '@/components/app/preparing-view';
import { ProgressIndicator } from '@/components/app/progress-indicator';
import {
  type InterviewSummary,
  type TranscriptEntry,
  SummaryView,
} from '@/components/app/summary-view';
import { type InterviewSetup, WelcomeView } from '@/components/app/welcome-view';

const MotionWelcomeView = motion.create(WelcomeView);
const MotionPreparingView = motion.create(PreparingView);
const MotionSessionView = motion.create(AgentSessionView_01);
const MotionSummaryView = motion.create(SummaryView);

const VIEW_MOTION_PROPS = {
  variants: {
    visible: {
      opacity: 1,
    },
    hidden: {
      opacity: 0,
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.5,
    ease: 'linear',
  },
};

// Matches the backend's InterviewState.max_questions default (see
// backend/my-agent/src/interview_flow.py) so the progress bar can show "question 1 of 8"
// immediately, before the first real interview-progress data message arrives.
const DEFAULT_MAX_QUESTIONS = 8;

interface QuestionProgress {
  questionsAsked: number;
  maxQuestions: number;
}

// The four screens of the interview flow. "preparing" covers the room-connection window
// between submitting the setup form and the agent actually being ready; it's a plain
// connection spinner (no wait on resume parsing -- see backend/my-agent/src/agent.py's
// _choose_question_bank, which deliberately doesn't wait for a late resume upload either).
type Phase = 'welcome' | 'preparing' | 'session' | 'summary';

interface ViewControllerProps {
  appConfig: AppConfig;
}

export function ViewController({ appConfig }: ViewControllerProps) {
  const { isConnected, start, end } = useSessionContext();
  const { resolvedTheme } = useTheme();
  const room = useRoomContext();
  const { send } = useChat();

  const [phase, setPhase] = useState<Phase>('welcome');
  const [progress, setProgress] = useState<QuestionProgress>({
    questionsAsked: 0,
    maxQuestions: DEFAULT_MAX_QUESTIONS,
  });
  const [summary, setSummary] = useState<InterviewSummary | null>(null);

  // Refs so data-channel/effect callbacks always see the latest value without having to
  // be re-subscribed on every change.
  const progressRef = useRef(progress);
  progressRef.current = progress;
  const sessionStartRef = useRef<number | null>(null);

  // Filled in when the candidate submits the setup form, then flushed once the room
  // connection is up (see the effect below) -- the resume/job title can't be sent to the
  // agent until the participant is actually connected.
  const pendingSetupRef = useRef<InterviewSetup | null>(null);
  const setupSentRef = useRef(false);

  // Filled in incrementally as each question is answered (see the "interview-transcript-entry"
  // listener below), so the interview's actual Q&A survives an early disconnect -- the full,
  // feedback-annotated transcript from "interview-complete" below still takes priority when the
  // interview finishes normally.
  const transcriptRef = useRef<TranscriptEntry[]>([]);

  const buildSummary = (
    transcript: TranscriptEntry[] = transcriptRef.current
  ): InterviewSummary => ({
    jobTitle: pendingSetupRef.current?.jobTitle ?? '',
    durationSeconds: sessionStartRef.current
      ? Math.round((Date.now() - sessionStartRef.current) / 1000)
      : 0,
    questionsAsked: progressRef.current.questionsAsked,
    maxQuestions: progressRef.current.maxQuestions,
    transcript,
  });

  // The backend publishes these two data messages from InterviewerAgent (see
  // backend/my-agent/src/agent.py's _publish_data/_publish_progress/_generate_and_publish_summary):
  // "interview-progress" after every question is answered, and "interview-complete" once the
  // agent wraps up (carrying the full question/answer/feedback transcript as its payload).
  useDataChannel('interview-progress', (message) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(message.payload));
      if (typeof data.questionsAsked === 'number' && typeof data.maxQuestions === 'number') {
        setProgress({ questionsAsked: data.questionsAsked, maxQuestions: data.maxQuestions });
      }
    } catch (error) {
      console.error('Failed to parse interview-progress message', error);
    }
  });

  // Lets the candidate know whether their uploaded resume actually got parsed (see
  // backend/my-agent/src/agent.py's _process_resume_upload, which publishes this once it
  // finishes trying) -- previously this failed silently, so a bad PDF (e.g. a scanned
  // image with no text layer) looked identical to a successful upload.
  useDataChannel('resume-status', (message) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(message.payload));
      if (data.status === 'success') {
        toast.success('Resume processed', {
          description: "We'll tailor your interview questions to it.",
        });
      } else if (data.status === 'error') {
        toast.error("Couldn't read your resume", {
          description: 'Continuing with generic interview questions instead.',
        });
      }
    } catch (error) {
      console.error('Failed to parse resume-status message', error);
    }
  });

  // Accumulates each answered question as it happens (see agent.py's _ask_and_record),
  // independently of "interview-progress" (which only carries counts) and
  // "interview-complete" (which only arrives at a normal finish).
  useDataChannel('interview-transcript-entry', (message) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(message.payload));
      if (typeof data.question === 'string' && typeof data.answer === 'string') {
        transcriptRef.current = [
          ...transcriptRef.current,
          { question: data.question, answer: data.answer, feedback: null },
        ];
      }
    } catch (error) {
      console.error('Failed to parse interview-transcript-entry message', error);
    }
  });

  useDataChannel('interview-complete', (message) => {
    let transcript: TranscriptEntry[] = transcriptRef.current;
    try {
      const data = JSON.parse(new TextDecoder().decode(message.payload));
      if (Array.isArray(data.transcript)) {
        transcript = data.transcript;
      }
    } catch (error) {
      console.error('Failed to parse interview-complete message', error);
    }
    setSummary(buildSummary(transcript));
    setPhase('summary');
  });

  const handleStartCall = async (setup: InterviewSetup) => {
    pendingSetupRef.current = setup;
    setupSentRef.current = false;
    transcriptRef.current = [];
    setPhase('preparing');
    try {
      await start();
    } catch (error) {
      console.error('Failed to start the interview session', error);
      toast.error('Could not start the interview', {
        description:
          'Check that your microphone is allowed for this site in your browser settings, then try again.',
      });
      setPhase('welcome');
    }
  };

  const handleRestart = () => {
    pendingSetupRef.current = null;
    setupSentRef.current = false;
    transcriptRef.current = [];
    setProgress({ questionsAsked: 0, maxQuestions: DEFAULT_MAX_QUESTIONS });
    sessionStartRef.current = null;
    setSummary(null);
    setPhase('welcome');
    if (isConnected) {
      end();
    }
  };

  // preparing -> session, once the room connection actually comes up
  useEffect(() => {
    if (isConnected && phase === 'preparing') {
      sessionStartRef.current = Date.now();
      setPhase('session');
    }
  }, [isConnected, phase]);

  // Safety net: if the room disconnects while still mid-interview (candidate hangs up
  // manually, or the agent/job crashes) rather than via the normal "interview-complete"
  // data message, still land on the summary screen instead of getting stuck.
  useEffect(() => {
    if (!isConnected && phase === 'session') {
      setSummary(buildSummary());
      setPhase('summary');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected, phase]);

  useEffect(() => {
    if (!isConnected || setupSentRef.current) return;
    const setup = pendingSetupRef.current;
    if (!setup) return;
    setupSentRef.current = true;

    void (async () => {
      // Send the job title first, before the resume upload. The candidate already gave us
      // their target role on the Welcome form, and the agent's system prompt (see
      // InterviewerAgent's instructions in backend/my-agent/src/agent.py) tells it to check
      // for an already-stated role and skip asking for one if it sees it -- but it can only
      // see it if this chat message actually lands before the agent's opening turn. The
      // resume file transfer is the slower of the two (it's a whole PDF), so sending it
      // first would needlessly delay the job title and risk the agent asking for the role
      // anyway. Sending the job title first costs us nothing: _choose_question_bank already
      // tolerates the resume arriving late by falling back to GENERIC_QUESTIONS.
      if (setup.jobTitle) {
        try {
          await send(`I'd like to practice for a ${setup.jobTitle} role.`);
        } catch (error) {
          console.error('Failed to send job title to the interview agent', error);
        }
      }
      if (setup.resumeFile) {
        try {
          await room.localParticipant.sendFile(setup.resumeFile, {
            topic: 'resume',
            mimeType: 'application/pdf',
          });
        } catch (error) {
          console.error('Failed to send resume to the interview agent', error);
        }
      }
    })();
  }, [isConnected, room, send]);

  return (
    <>
      {phase === 'session' && (
        <ProgressIndicator
          questionsAsked={progress.questionsAsked}
          maxQuestions={progress.maxQuestions}
        />
      )}
      <AnimatePresence mode="wait">
        {/* Welcome view */}
        {phase === 'welcome' && (
          <MotionWelcomeView
            key="welcome"
            {...VIEW_MOTION_PROPS}
            startButtonText={appConfig.startButtonText}
            onStartCall={handleStartCall}
          />
        )}
        {/* Preparing view: connecting to the room/agent */}
        {phase === 'preparing' && <MotionPreparingView key="preparing" {...VIEW_MOTION_PROPS} />}
        {/* Session view */}
        {phase === 'session' && (
          <MotionSessionView
            key="session-view"
            {...VIEW_MOTION_PROPS}
            supportsChatInput={appConfig.supportsChatInput}
            supportsVideoInput={appConfig.supportsVideoInput}
            supportsScreenShare={appConfig.supportsScreenShare}
            isPreConnectBufferEnabled={appConfig.isPreConnectBufferEnabled}
            audioVisualizerType={appConfig.audioVisualizerType}
            audioVisualizerColor={
              resolvedTheme === 'dark'
                ? appConfig.audioVisualizerColorDark
                : appConfig.audioVisualizerColor
            }
            audioVisualizerColorShift={appConfig.audioVisualizerColorShift}
            audioVisualizerBarCount={appConfig.audioVisualizerBarCount}
            audioVisualizerGridRowCount={appConfig.audioVisualizerGridRowCount}
            audioVisualizerGridColumnCount={appConfig.audioVisualizerGridColumnCount}
            audioVisualizerRadialBarCount={appConfig.audioVisualizerRadialBarCount}
            audioVisualizerRadialRadius={appConfig.audioVisualizerRadialRadius}
            audioVisualizerWaveLineWidth={appConfig.audioVisualizerWaveLineWidth}
            className="fixed inset-0"
          />
        )}
        {/* Summary view */}
        {phase === 'summary' && summary && (
          <MotionSummaryView
            key="summary"
            {...VIEW_MOTION_PROPS}
            summary={summary}
            onRestart={handleRestart}
          />
        )}
      </AnimatePresence>
    </>
  );
}
