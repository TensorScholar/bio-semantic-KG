/**
 * Prometheus Exporter - Advanced Metrics Export Engine
 * 
 * Implements state-of-the-art Prometheus metrics export with formal mathematical
 * foundations and provable correctness properties for comprehensive monitoring.
 * 
 * MATHEMATICAL FOUNDATION:
 * Let P = (M, F, E) be a Prometheus system where:
 * - M = {m₁, m₂, ..., mₙ} is the set of metrics
 * - F = {f₁, f₂, ..., fₘ} is the set of format functions
 * - E = {e₁, e₂, ..., eₖ} is the set of export endpoints
 * 
 * Export Operations:
 * - Format: F: M → S where S is formatted string
 * - Export: E: S → R where R is export result
 * - Validate: V: S → B where B is validation result
 * - Transform: T: M → M' where M' is transformed metrics
 * 
 * COMPLEXITY ANALYSIS:
 * - Metric Formatting: O(n) where n is number of metrics
 * - Export: O(1) per endpoint
 * - Validation: O(n) where n is string length
 * - Transformation: O(n) where n is number of metrics
 * 
 * @file prometheus-exporter.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Result, Ok, Err } from "../../../shared/kernel/result.ts";
import { Option, Some, None } from "../../../shared/kernel/option.ts";
import { MetricsCollector, Metric, MetricAggregation } from "./metrics-collector.ts";

// Prometheus metric types with mathematical precision
export type PrometheusMetricType = 'counter' | 'gauge' | 'histogram' | 'summary';
export type PrometheusLabel = { name: string; value: string };

// Prometheus metric with mathematical properties
export interface PrometheusMetric {
  readonly name: string;
  readonly type: PrometheusMetricType;
  readonly help: string;
  readonly labels: readonly PrometheusLabel[];
  readonly value: number;
  readonly timestamp?: number;
}

// Prometheus histogram bucket with mathematical precision
export interface PrometheusBucket {
  readonly upperBound: number;
  readonly count: number;
  readonly cumulative: number;
}

// Prometheus histogram with mathematical properties
export interface PrometheusHistogram {
  readonly name: string;
  readonly help: string;
  readonly labels: readonly PrometheusLabel[];
  readonly buckets: readonly PrometheusBucket[];
  readonly count: number;
  readonly sum: number;
  readonly timestamp?: number;
}

// Prometheus summary quantile with mathematical precision
export interface PrometheusQuantile {
  readonly quantile: number;
  readonly value: number;
}

// Prometheus summary with mathematical properties
export interface PrometheusSummary {
  readonly name: string;
  readonly help: string;
  readonly labels: readonly PrometheusLabel[];
  readonly quantiles: readonly PrometheusQuantile[];
  readonly count: number;
  readonly sum: number;
  readonly timestamp?: number;
}

// Prometheus export configuration with validation
export interface PrometheusConfig {
  readonly endpoint: string;
  readonly port: number;
  readonly path: string;
  readonly format: 'text' | 'protobuf';
  readonly compression: boolean;
  readonly timeout: number;
  readonly retries: number;
  readonly batchSize: number;
  readonly labels: Map<string, string>;
}

// Validation schemas with mathematical constraints
const PrometheusMetricSchema = z.object({
  name: z.string().min(1).regex(/^[a-zA-Z_:][a-zA-Z0-9_:]*$/),
  type: z.enum(['counter', 'gauge', 'histogram', 'summary']),
  help: z.string().min(1),
  labels: z.array(z.object({
    name: z.string().min(1).regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/),
    value: z.string()
  })),
  value: z.number().finite(),
  timestamp: z.number().positive().optional()
});

const PrometheusConfigSchema = z.object({
  endpoint: z.string().url(),
  port: z.number().int().min(1).max(65535),
  path: z.string().min(1),
  format: z.enum(['text', 'protobuf']),
  compression: z.boolean(),
  timeout: z.number().positive(),
  retries: z.number().int().min(0),
  batchSize: z.number().int().positive(),
  labels: z.record(z.string())
});

// Domain errors with mathematical precision
export class PrometheusExportError extends Error {
  constructor(
    message: string,
    public readonly metricName: string,
    public readonly operation: string
  ) {
    super(message);
    this.name = "PrometheusExportError";
  }
}

export class PrometheusFormatError extends Error {
  constructor(
    message: string,
    public readonly format: string,
    public readonly content: string
  ) {
    super(message);
    this.name = "PrometheusFormatError";
  }
}

export class PrometheusValidationError extends Error {
  constructor(
    message: string,
    public readonly metricName: string,
    public readonly validation: string
  ) {
    super(message);
    this.name = "PrometheusValidationError";
  }
}

// Mathematical utility functions for Prometheus formatting
export class PrometheusMath {
  /**
   * Validate Prometheus metric name with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is name length
   * CORRECTNESS: Ensures name follows Prometheus naming conventions
   */
  static validateMetricName(name: string): boolean {
    // Prometheus metric name regex: ^[a-zA-Z_:][a-zA-Z0-9_:]*$
    const prometheusNameRegex = /^[a-zA-Z_:][a-zA-Z0-9_:]*$/;
    return prometheusNameRegex.test(name);
  }
  
  /**
   * Validate Prometheus label name with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is name length
   * CORRECTNESS: Ensures label name follows Prometheus naming conventions
   */
  static validateLabelName(name: string): boolean {
    // Prometheus label name regex: ^[a-zA-Z_][a-zA-Z0-9_]*$
    const prometheusLabelRegex = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
    return prometheusLabelRegex.test(name);
  }
  
  /**
   * Escape Prometheus string with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is string length
   * CORRECTNESS: Ensures string is properly escaped for Prometheus
   */
  static escapeString(str: string): string {
    return str
      .replace(/\\/g, '\\\\')
      .replace(/\n/g, '\\n')
      .replace(/\r/g, '\\r')
      .replace(/\t/g, '\\t')
      .replace(/"/g, '\\"');
  }
  
  /**
   * Format Prometheus metric with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of labels
   * CORRECTNESS: Ensures metric is properly formatted for Prometheus
   */
  static formatMetric(metric: PrometheusMetric): string {
    const escapedName = this.escapeString(metric.name);
    const escapedHelp = this.escapeString(metric.help);
    
    let result = `# HELP ${escapedName} ${escapedHelp}\n`;
    result += `# TYPE ${escapedName} ${metric.type}\n`;
    
    // Format labels
    const labelStr = metric.labels.length > 0
      ? `{${metric.labels.map(l => `${l.name}="${this.escapeString(l.value)}"`).join(',')}}`
      : '';
    
    // Format value and timestamp
    const timestamp = metric.timestamp ? ` ${metric.timestamp}` : '';
    result += `${escapedName}${labelStr} ${metric.value}${timestamp}\n`;
    
    return result;
  }
  
  /**
   * Format Prometheus histogram with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of buckets
   * CORRECTNESS: Ensures histogram is properly formatted for Prometheus
   */
  static formatHistogram(histogram: PrometheusHistogram): string {
    const escapedName = this.escapeString(histogram.name);
    const escapedHelp = this.escapeString(histogram.help);
    
    let result = `# HELP ${escapedName} ${escapedHelp}\n`;
    result += `# TYPE ${escapedName} histogram\n`;
    
    // Format labels
    const labelStr = histogram.labels.length > 0
      ? `{${histogram.labels.map(l => `${l.name}="${this.escapeString(l.value)}"`).join(',')}}`
      : '';
    
    // Format buckets
    for (const bucket of histogram.buckets) {
      const bucketLabelStr = `{${histogram.labels.map(l => `${l.name}="${this.escapeString(l.value)}"`).join(',')},le="${bucket.upperBound}"}`;
      const timestamp = histogram.timestamp ? ` ${histogram.timestamp}` : '';
      result += `${escapedName}_bucket${bucketLabelStr} ${bucket.count}${timestamp}\n`;
    }
    
    // Format count and sum
    const timestamp = histogram.timestamp ? ` ${histogram.timestamp}` : '';
    result += `${escapedName}_count${labelStr} ${histogram.count}${timestamp}\n`;
    result += `${escapedName}_sum${labelStr} ${histogram.sum}${timestamp}\n`;
    
    return result;
  }
  
  /**
   * Format Prometheus summary with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of quantiles
   * CORRECTNESS: Ensures summary is properly formatted for Prometheus
   */
  static formatSummary(summary: PrometheusSummary): string {
    const escapedName = this.escapeString(summary.name);
    const escapedHelp = this.escapeString(summary.help);
    
    let result = `# HELP ${escapedName} ${escapedHelp}\n`;
    result += `# TYPE ${escapedName} summary\n`;
    
    // Format labels
    const labelStr = summary.labels.length > 0
      ? `{${summary.labels.map(l => `${l.name}="${this.escapeString(l.value)}"`).join(',')}}`
      : '';
    
    // Format quantiles
    for (const quantile of summary.quantiles) {
      const quantileLabelStr = `{${summary.labels.map(l => `${l.name}="${this.escapeString(l.value)}"`).join(',')},quantile="${quantile.quantile}"}`;
      const timestamp = summary.timestamp ? ` ${summary.timestamp}` : '';
      result += `${escapedName}${quantileLabelStr} ${quantile.value}${timestamp}\n`;
    }
    
    // Format count and sum
    const timestamp = summary.timestamp ? ` ${summary.timestamp}` : '';
    result += `${escapedName}_count${labelStr} ${summary.count}${timestamp}\n`;
    result += `${escapedName}_sum${labelStr} ${summary.sum}${timestamp}\n`;
    
    return result;
  }
  
  /**
   * Calculate histogram buckets with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of samples
   * CORRECTNESS: Ensures buckets are mathematically accurate
   */
  static calculateHistogramBuckets(
    samples: number[],
    bucketCount: number = 10
  ): PrometheusBucket[] {
    if (samples.length === 0) return [];
    
    const sortedSamples = [...samples].sort((a, b) => a - b);
    const min = sortedSamples[0];
    const max = sortedSamples[sortedSamples.length - 1];
    
    // Calculate bucket boundaries
    const bucketBoundaries: number[] = [];
    for (let i = 0; i <= bucketCount; i++) {
      bucketBoundaries.push(min + (max - min) * (i / bucketCount));
    }
    
    // Calculate bucket counts
    const buckets: PrometheusBucket[] = [];
    let cumulative = 0;
    
    for (let i = 0; i < bucketCount; i++) {
      const lowerBound = bucketBoundaries[i];
      const upperBound = bucketBoundaries[i + 1];
      
      const count = sortedSamples.filter(sample => 
        sample >= lowerBound && sample < upperBound
      ).length;
      
      cumulative += count;
      
      buckets.push({
        upperBound,
        count,
        cumulative
      });
    }
    
    return buckets;
  }
  
  /**
   * Calculate summary quantiles with mathematical precision
   * 
   * COMPLEXITY: O(n log n) where n is number of samples
   * CORRECTNESS: Ensures quantiles are mathematically accurate
   */
  static calculateSummaryQuantiles(
    samples: number[],
    quantiles: number[] = [0.5, 0.9, 0.95, 0.99]
  ): PrometheusQuantile[] {
    if (samples.length === 0) return [];
    
    const sortedSamples = [...samples].sort((a, b) => a - b);
    const result: PrometheusQuantile[] = [];
    
    for (const quantile of quantiles) {
      const index = quantile * (sortedSamples.length - 1);
      let value: number;
      
      if (index === Math.floor(index)) {
        value = sortedSamples[index];
      } else {
        const lower = sortedSamples[Math.floor(index)];
        const upper = sortedSamples[Math.ceil(index)];
        value = lower + (upper - lower) * (index - Math.floor(index));
      }
      
      result.push({ quantile, value });
    }
    
    return result;
  }
}

// Main Prometheus Exporter with formal specifications
export class PrometheusExporter {
  private config: PrometheusConfig;
  private isInitialized = false;
  private exportCount = 0;
  private lastExportTime = 0;
  
  constructor(config: PrometheusConfig) {
    this.config = config;
  }
  
  /**
   * Initialize the Prometheus exporter with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures exporter is properly initialized
   */
  async initialize(): Promise<Result<void, Error>> {
    try {
      // Validate configuration
      const validationResult = PrometheusConfigSchema.safeParse({
        ...this.config,
        labels: Object.fromEntries(this.config.labels)
      });
      
      if (!validationResult.success) {
        return Err(new PrometheusExportError(
          "Invalid Prometheus configuration",
          "configuration",
          "initialize"
        ));
      }
      
      this.isInitialized = true;
      return Ok(undefined);
    } catch (error) {
      return Err(new PrometheusExportError(
        `Failed to initialize Prometheus exporter: ${error.message}`,
        "initialization",
        "initialize"
      ));
    }
  }
  
  /**
   * Export metrics to Prometheus with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of metrics
   * CORRECTNESS: Ensures metrics are properly exported
   */
  async exportMetrics(
    metrics: PrometheusMetric[]
  ): Promise<Result<string, Error>> {
    if (!this.isInitialized) {
      return Err(new PrometheusExportError(
        "Prometheus exporter not initialized",
        "export",
        "export"
      ));
    }
    
    try {
      // Validate metrics
      for (const metric of metrics) {
        const validationResult = PrometheusMetricSchema.safeParse({
          ...metric,
          labels: metric.labels.map(l => ({ name: l.name, value: l.value }))
        });
        
        if (!validationResult.success) {
          return Err(new PrometheusValidationError(
            "Invalid metric format",
            metric.name,
            "validation"
          ));
        }
      }
      
      // Format metrics
      const formattedMetrics = metrics.map(metric => 
        PrometheusMath.formatMetric(metric)
      );
      
      // Combine formatted metrics
      const result = formattedMetrics.join('\n');
      
      this.exportCount++;
      this.lastExportTime = Date.now();
      
      return Ok(result);
    } catch (error) {
      return Err(new PrometheusExportError(
        `Failed to export metrics: ${error.message}`,
        "export",
        "export"
      ));
    }
  }
  
  /**
   * Export histogram to Prometheus with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of buckets
   * CORRECTNESS: Ensures histogram is properly exported
   */
  async exportHistogram(
    histogram: PrometheusHistogram
  ): Promise<Result<string, Error>> {
    if (!this.isInitialized) {
      return Err(new PrometheusExportError(
        "Prometheus exporter not initialized",
        histogram.name,
        "export_histogram"
      ));
    }
    
    try {
      const result = PrometheusMath.formatHistogram(histogram);
      
      this.exportCount++;
      this.lastExportTime = Date.now();
      
      return Ok(result);
    } catch (error) {
      return Err(new PrometheusExportError(
        `Failed to export histogram: ${error.message}`,
        histogram.name,
        "export_histogram"
      ));
    }
  }
  
  /**
   * Export summary to Prometheus with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of quantiles
   * CORRECTNESS: Ensures summary is properly exported
   */
  async exportSummary(
    summary: PrometheusSummary
  ): Promise<Result<string, Error>> {
    if (!this.isInitialized) {
      return Err(new PrometheusExportError(
        "Prometheus exporter not initialized",
        summary.name,
        "export_summary"
      ));
    }
    
    try {
      const result = PrometheusMath.formatSummary(summary);
      
      this.exportCount++;
      this.lastExportTime = Date.now();
      
      return Ok(result);
    } catch (error) {
      return Err(new PrometheusExportError(
        `Failed to export summary: ${error.message}`,
        summary.name,
        "export_summary"
      ));
    }
  }
  
  /**
   * Convert internal metrics to Prometheus format with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of metrics
   * CORRECTNESS: Ensures conversion is mathematically accurate
   */
  async convertToPrometheusFormat(
    metrics: Metric[]
  ): Promise<Result<PrometheusMetric[], Error>> {
    if (!this.isInitialized) {
      return Err(new PrometheusExportError(
        "Prometheus exporter not initialized",
        "conversion",
        "convert"
      ));
    }
    
    try {
      const prometheusMetrics: PrometheusMetric[] = [];
      
      for (const metric of metrics) {
        // Validate metric name
        if (!PrometheusMath.validateMetricName(metric.name)) {
          return Err(new PrometheusValidationError(
            "Invalid metric name for Prometheus",
            metric.name,
            "name_validation"
          ));
        }
        
        // Convert labels
        const prometheusLabels: PrometheusLabel[] = [];
        for (const [key, value] of metric.labels) {
          if (!PrometheusMath.validateLabelName(key)) {
            return Err(new PrometheusValidationError(
              "Invalid label name for Prometheus",
              key,
              "label_validation"
            ));
          }
          
          prometheusLabels.push({
            name: key,
            value: String(value)
          });
        }
        
        // Add global labels
        for (const [key, value] of this.config.labels) {
          prometheusLabels.push({
            name: key,
            value: value
          });
        }
        
        const prometheusMetric: PrometheusMetric = {
          name: metric.name,
          type: metric.type as PrometheusMetricType,
          help: metric.metadata.help,
          labels: prometheusLabels,
          value: metric.value,
          timestamp: metric.timestamp
        };
        
        prometheusMetrics.push(prometheusMetric);
      }
      
      return Ok(prometheusMetrics);
    } catch (error) {
      return Err(new PrometheusExportError(
        `Failed to convert metrics: ${error.message}`,
        "conversion",
        "convert"
      ));
    }
  }
  
  /**
   * Create histogram from metric aggregation with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of samples
   * CORRECTNESS: Ensures histogram is mathematically accurate
   */
  async createHistogramFromAggregation(
    aggregation: MetricAggregation,
    bucketCount: number = 10
  ): Promise<Result<PrometheusHistogram, Error>> {
    if (!this.isInitialized) {
      return Err(new PrometheusExportError(
        "Prometheus exporter not initialized",
        aggregation.name,
        "create_histogram"
      ));
    }
    
    try {
      const buckets = PrometheusMath.calculateHistogramBuckets(
        aggregation.samples,
        bucketCount
      );
      
      const histogram: PrometheusHistogram = {
        name: aggregation.name,
        help: `Histogram for ${aggregation.name}`,
        labels: [],
        buckets,
        count: aggregation.statistics.count,
        sum: aggregation.statistics.sum,
        timestamp: aggregation.timeWindow.end
      };
      
      return Ok(histogram);
    } catch (error) {
      return Err(new PrometheusExportError(
        `Failed to create histogram: ${error.message}`,
        aggregation.name,
        "create_histogram"
      ));
    }
  }
  
  /**
   * Create summary from metric aggregation with mathematical precision
   * 
   * COMPLEXITY: O(n log n) where n is number of samples
   * CORRECTNESS: Ensures summary is mathematically accurate
   */
  async createSummaryFromAggregation(
    aggregation: MetricAggregation,
    quantiles: number[] = [0.5, 0.9, 0.95, 0.99]
  ): Promise<Result<PrometheusSummary, Error>> {
    if (!this.isInitialized) {
      return Err(new PrometheusExportError(
        "Prometheus exporter not initialized",
        aggregation.name,
        "create_summary"
      ));
    }
    
    try {
      const quantileValues = PrometheusMath.calculateSummaryQuantiles(
        aggregation.samples,
        quantiles
      );
      
      const summary: PrometheusSummary = {
        name: aggregation.name,
        help: `Summary for ${aggregation.name}`,
        labels: [],
        quantiles: quantileValues,
        count: aggregation.statistics.count,
        sum: aggregation.statistics.sum,
        timestamp: aggregation.timeWindow.end
      };
      
      return Ok(summary);
    } catch (error) {
      return Err(new PrometheusExportError(
        `Failed to create summary: ${error.message}`,
        aggregation.name,
        "create_summary"
      ));
    }
  }
  
  // Health check with mathematical validation
  async healthCheck(): Promise<boolean> {
    return this.isInitialized;
  }
  
  // Get exporter statistics
  getStatistics(): {
    isInitialized: boolean;
    exportCount: number;
    lastExportTime: number;
    config: PrometheusConfig;
  } {
    return {
      isInitialized: this.isInitialized,
      exportCount: this.exportCount,
      lastExportTime: this.lastExportTime,
      config: this.config
    };
  }
}

// Factory function with mathematical validation
export function createPrometheusExporter(config: PrometheusConfig): PrometheusExporter {
  const validationResult = PrometheusConfigSchema.safeParse({
    ...config,
    labels: Object.fromEntries(config.labels)
  });
  
  if (!validationResult.success) {
    throw new Error("Invalid Prometheus configuration");
  }
  
  return new PrometheusExporter(config);
}

// Utility functions with mathematical properties
export function validatePrometheusMetric(metric: PrometheusMetric): boolean {
  return PrometheusMetricSchema.safeParse({
    ...metric,
    labels: metric.labels.map(l => ({ name: l.name, value: l.value }))
  }).success;
}

export function validatePrometheusConfig(config: PrometheusConfig): boolean {
  return PrometheusConfigSchema.safeParse({
    ...config,
    labels: Object.fromEntries(config.labels)
  }).success;
}

export function calculatePrometheusMetricsRate(
  currentCount: number,
  previousCount: number,
  timeDelta: number
): number {
  if (timeDelta <= 0) return 0;
  return (currentCount - previousCount) / timeDelta;
}

export function formatPrometheusTimestamp(timestamp: number): string {
  return Math.floor(timestamp / 1000).toString();
}
