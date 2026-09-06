'use client';

import { useEffect, useRef } from 'react';
import { useTheme } from 'next-themes';
import { AnimatePresence, motion } from 'motion/react';
import { useChat, useRoomContext, useSessionContext } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AgentSessionView_01 } from '@/components/agents-ui/blocks/agent-session-view-01';
import { type InterviewSetup, WelcomeView } from '@/components/app/welcome-view';

const MotionWelcomeView = motion.create(WelcomeView);
const MotionSessionView = motion.create(AgentSessionView_01);

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

interface ViewControllerProps {
  appConfig: AppConfig;
}

export function ViewController({ appConfig }: ViewControllerProps) {
  const { isConnected, start } = useSessionContext();
  const { resolvedTheme } = useTheme();
  const room = useRoomContext();
  const { send } = useChat();

  // Filled in when the candidate submits the setup form, then flushed once the room
  // connection is up (see the effect below) -- the resume/job title can't be sent to the
  // agent until the participant is actually connected.
  const pendingSetupRef = useRef<InterviewSetup | null>(null);
  const setupSentRef = useRef(false);

  const handleStartCall = (setup: InterviewSetup) => {
    pendingSetupRef.current = setup;
    setupSentRef.current = false;
    start();
  };

  useEffect(() => {
    if (!isConnected || setupSentRef.current) return;
    const setup = pendingSetupRef.current;
    if (!setup) return;
    setupSentRef.current = true;

    void (async () => {
      // Send the resume first and await full delivery before the job title message --
      // the agent uses whichever resume text has already arrived by the time it processes
      // the job title, so sending resume-before-title gives it the best chance of being
      // ready in time (see backend/my-agent/src/agent.py's _choose_question_bank).
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
      if (setup.jobTitle) {
        try {
          await send(`I'd like to practice for a ${setup.jobTitle} role.`);
        } catch (error) {
          console.error('Failed to send job title to the interview agent', error);
        }
      }
    })();
  }, [isConnected, room, send]);

  return (
    <AnimatePresence mode="wait">
      {/* Welcome view */}
      {!isConnected && (
        <MotionWelcomeView
          key="welcome"
          {...VIEW_MOTION_PROPS}
          startButtonText={appConfig.startButtonText}
          onStartCall={handleStartCall}
        />
      )}
      {/* Session view */}
      {isConnected && (
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
    </AnimatePresence>
  );
}
