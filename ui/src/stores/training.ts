/**
 * Training state management store using Zustand
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { TrainingProgressData } from '../types/websocket';

// Training job interface
export interface TrainingJob {
  jobId: string;
  status: string;
  symbols: string[];
  timeframe: string;
  phase: 'transformer' | 'agent';
  progress: {
    currentEpoch?: number;
    totalEpochs?: number;
    percent: number;
    currentTimestep?: number;
    totalTimesteps?: number;
  };
  metrics: {
    trainLoss?: number;
    valLoss?: number;
    learningRate?: number;
  };
  timing: {
    startedAt?: string;
    elapsedSeconds: number;
    estimatedRemainingSeconds?: number;
  };
  createdAt: string;
  lossHistory: {
    epoch: number;
    trainLoss: number;
    valLoss?: number;
  }[];
}

// Training configuration draft
export interface TrainingConfigDraft {
  symbols: string[];
  timeframe: string;
  multiTimeframe: boolean;
  additionalTimeframes: string[];
  bars: number;
  epochs: number;
  timesteps: number;
  modelDir: string;
  transformer: {
    dModel: number;
    nHeads: number;
    nEncoderLayers: number;
    dropout: number;
    learningRate: number;
  };
  ppo: {
    learningRate: number;
    gamma: number;
    clipEpsilon: number;
    entropyCoefficient: number;
  };
}

// Default configuration
const defaultConfig: TrainingConfigDraft = {
  symbols: ['EURUSD'],
  timeframe: '1h',
  multiTimeframe: false,
  additionalTimeframes: [],
  bars: 50000,
  epochs: 100,
  timesteps: 1000000,
  modelDir: './saved_models',
  transformer: {
    dModel: 128,
    nHeads: 8,
    nEncoderLayers: 4,
    dropout: 0.1,
    learningRate: 0.0001,
  },
  ppo: {
    learningRate: 0.0003,
    gamma: 0.99,
    clipEpsilon: 0.2,
    entropyCoefficient: 0.01,
  },
};

interface TrainingState {
  // Active jobs being monitored
  activeJobs: Map<string, TrainingJob>;
  // Currently selected job for detailed view
  selectedJobId: string | null;
  // Configuration draft for new training runs
  configDraft: TrainingConfigDraft;
  // Loading state
  isLoading: boolean;
}

interface TrainingActions {
  // Job management
  updateJob: (jobId: string, data: Partial<TrainingJob>) => void;
  updateJobFromProgress: (data: TrainingProgressData) => void;
  removeJob: (jobId: string) => void;
  selectJob: (jobId: string | null) => void;
  clearCompletedJobs: () => void;

  // Configuration
  setConfigDraft: (config: Partial<TrainingConfigDraft>) => void;
  resetConfigDraft: () => void;

  // Loading
  setLoading: (loading: boolean) => void;

  // Loss history
  appendLossHistory: (jobId: string, entry: { epoch: number; trainLoss: number; valLoss?: number }) => void;
}

export const useTrainingStore = create<TrainingState & TrainingActions>()(
  devtools(
    persist(
      (set) => ({
        // Initial state
        activeJobs: new Map(),
        selectedJobId: null,
        configDraft: defaultConfig,
        isLoading: false,

        // Actions
        updateJob: (jobId, data) =>
          set((state) => {
            const newJobs = new Map(state.activeJobs);
            const existing = newJobs.get(jobId);
            if (existing) {
              newJobs.set(jobId, { ...existing, ...data });
            } else {
              newJobs.set(jobId, {
                jobId,
                status: 'running',
                symbols: [],
                timeframe: '1h',
                phase: 'transformer',
                progress: { percent: 0 },
                metrics: {},
                timing: { elapsedSeconds: 0 },
                createdAt: new Date().toISOString(),
                lossHistory: [],
                ...data,
              } as TrainingJob);
            }
            return { activeJobs: newJobs };
          }),

        updateJobFromProgress: (data) =>
          set((state) => {
            const newJobs = new Map(state.activeJobs);
            const existing = newJobs.get(data.jobId);

            const updatedJob: TrainingJob = {
              jobId: data.jobId,
              status: data.status,
              symbols: existing?.symbols || [],
              timeframe: existing?.timeframe || '1h',
              phase: data.phase || 'transformer',
              progress: {
                currentEpoch: data.epoch || data.progress?.currentEpoch,
                totalEpochs: data.totalEpochs || data.progress?.totalEpochs,
                percent: data.progress?.percent || 0,
                currentTimestep: data.progress?.currentTimestep,
                totalTimesteps: data.progress?.totalTimesteps,
              },
              metrics: {
                trainLoss: data.trainLoss,
                valLoss: data.valLoss,
                learningRate: data.learningRate,
              },
              timing: {
                startedAt: existing?.timing.startedAt,
                elapsedSeconds: data.elapsedSeconds || 0,
                estimatedRemainingSeconds: data.estimatedRemainingSeconds,
              },
              createdAt: existing?.createdAt || new Date().toISOString(),
              lossHistory: existing?.lossHistory || [],
            };

            // Append to loss history if we have a new epoch with loss data
            if (data.trainLoss !== undefined && data.epoch !== undefined) {
              const lastEntry = updatedJob.lossHistory[updatedJob.lossHistory.length - 1];
              if (!lastEntry || lastEntry.epoch !== data.epoch) {
                updatedJob.lossHistory = [
                  ...updatedJob.lossHistory,
                  {
                    epoch: data.epoch,
                    trainLoss: data.trainLoss,
                    valLoss: data.valLoss,
                  },
                ];
              }
            }

            newJobs.set(data.jobId, updatedJob);
            return { activeJobs: newJobs };
          }),

        removeJob: (jobId) =>
          set((state) => {
            const newJobs = new Map(state.activeJobs);
            newJobs.delete(jobId);
            return {
              activeJobs: newJobs,
              selectedJobId: state.selectedJobId === jobId ? null : state.selectedJobId,
            };
          }),

        selectJob: (jobId) => set({ selectedJobId: jobId }),

        clearCompletedJobs: () =>
          set((state) => {
            const newJobs = new Map(state.activeJobs);
            for (const [jobId, job] of newJobs) {
              if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') {
                newJobs.delete(jobId);
              }
            }
            return { activeJobs: newJobs };
          }),

        setConfigDraft: (config) =>
          set((state) => ({
            configDraft: { ...state.configDraft, ...config },
          })),

        resetConfigDraft: () => set({ configDraft: defaultConfig }),

        setLoading: (loading) => set({ isLoading: loading }),

        appendLossHistory: (jobId, entry) =>
          set((state) => {
            const newJobs = new Map(state.activeJobs);
            const job = newJobs.get(jobId);
            if (job) {
              newJobs.set(jobId, {
                ...job,
                lossHistory: [...job.lossHistory, entry],
              });
            }
            return { activeJobs: newJobs };
          }),
      }),
      {
        name: 'leap-training-storage',
        // Only persist configDraft, not active jobs
        partialize: (state) => ({ configDraft: state.configDraft }),
      }
    ),
    { name: 'TrainingStore' }
  )
);
