/**
 * Extraction Port - Interface for Extraction Operations
 * 
 * Defines the contract for extraction operations following
 * Hexagonal Architecture principles with mathematical precision.
 * 
 * @file extraction.port.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { Result } from "../../../shared/kernel/result.ts";
import { MedicalClinic } from "../../../core/entities/medical-clinic.ts";
import { MedicalProcedure } from "../../../core/entities/medical-procedure.ts";

// Mathematical type definitions
export type ExtractionId = string;
export type SourceId = string;
export type ExtractionStatus = 'pending' | 'processing' | 'completed' | 'failed';

// Extraction entities with mathematical properties
export interface ExtractionSource {
  readonly id: SourceId;
  readonly url: string;
  readonly type: 'website' | 'api' | 'document' | 'database';
  readonly priority: number;
  readonly metadata: {
    readonly created: Date;
    readonly updated: Date;
    readonly lastExtracted: Date;
    readonly extractionCount: number;
    readonly successRate: number;
  };
}

export interface ExtractionJob {
  readonly id: ExtractionId;
  readonly sourceId: SourceId;
  readonly parserId: string;
  readonly status: ExtractionStatus;
  readonly config: ExtractionConfig;
  readonly metadata: {
    readonly created: Date;
    readonly started: Date;
    readonly completed?: Date;
    readonly duration?: number;
    readonly error?: string;
  };
}

export interface ExtractionConfig {
  readonly maxRetries: number;
  readonly timeout: number;
  readonly parallel: boolean;
  readonly filters: string[];
  readonly selectors: Record<string, string>;
  readonly nlpEnabled: boolean;
  readonly knowledgeGraphEnabled: boolean;
}

export interface ExtractionResult {
  readonly id: string;
  readonly jobId: ExtractionId;
  readonly clinics: MedicalClinic[];
  readonly procedures: MedicalProcedure[];
  readonly metadata: {
    readonly extracted: Date;
    readonly processingTime: number;
    readonly quality: number;
    readonly confidence: number;
  };
}

// Port interface for extraction operations
export interface ExtractionPort {
  /**
   * Add extraction source
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures source is properly added
   */
  addSource(source: ExtractionSource): Promise<Result<void, Error>>;
  
  /**
   * Get extraction source by ID
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures source is properly retrieved
   */
  getSource(sourceId: SourceId): Promise<Result<ExtractionSource, Error>>;
  
  /**
   * List all extraction sources
   * 
   * COMPLEXITY: O(n) where n is number of sources
   * CORRECTNESS: Ensures all sources are properly listed
   */
  listSources(): Promise<Result<ExtractionSource[], Error>>;
  
  /**
   * Start extraction job
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures job is properly started
   */
  startExtraction(
    sourceId: SourceId,
    parserId: string,
    config: ExtractionConfig
  ): Promise<Result<ExtractionJob, Error>>;
  
  /**
   * Get extraction job by ID
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures job is properly retrieved
   */
  getJob(jobId: ExtractionId): Promise<Result<ExtractionJob, Error>>;
  
  /**
   * List all extraction jobs
   * 
   * COMPLEXITY: O(n) where n is number of jobs
   * CORRECTNESS: Ensures all jobs are properly listed
   */
  listJobs(): Promise<Result<ExtractionJob[], Error>>;
  
  /**
   * Get extraction result by job ID
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures result is properly retrieved
   */
  getResult(jobId: ExtractionId): Promise<Result<ExtractionResult, Error>>;
  
  /**
   * List all extraction results
   * 
   * COMPLEXITY: O(n) where n is number of results
   * CORRECTNESS: Ensures all results are properly listed
   */
  listResults(): Promise<Result<ExtractionResult[], Error>>;
  
  /**
   * Cancel extraction job
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures job is properly cancelled
   */
  cancelJob(jobId: ExtractionId): Promise<Result<void, Error>>;
  
  /**
   * Get extraction statistics
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures statistics are mathematically accurate
   */
  getStatistics(): Promise<Result<ExtractionStatistics, Error>>;
}

// Extraction statistics with mathematical precision
export interface ExtractionStatistics {
  readonly totalSources: number;
  readonly activeSources: number;
  readonly totalJobs: number;
  readonly completedJobs: number;
  readonly failedJobs: number;
  readonly totalResults: number;
  readonly averageProcessingTime: number;
  readonly successRate: number;
  readonly qualityScore: number;
  readonly confidenceScore: number;
  readonly metadata: {
    readonly calculated: Date;
    readonly period: string;
  };
}
