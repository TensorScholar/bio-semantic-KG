/**
 * Rating Value Object - Immutable Rating System
 * 
 * Represents a rating with confidence intervals and validation.
 * Implements value object pattern with statistical rigor.
 * 
 * @file rating.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Branded } from "../../shared/types/branded.ts";

// Branded types for type safety
export type RatingValue = Branded<number, "RatingValue">;
export type ConfidenceInterval = Branded<number, "ConfidenceInterval">;
export type ReviewCount = Branded<number, "ReviewCount">;

// Rating scale configuration
export const RATING_SCALES = {
  FIVE_STAR: { min: 1, max: 5, step: 0.1, name: "5-Star Scale" },
  TEN_POINT: { min: 1, max: 10, step: 0.1, name: "10-Point Scale" },
  PERCENTAGE: { min: 0, max: 100, step: 1, name: "Percentage Scale" },
  LIKERT: { min: 1, max: 7, step: 1, name: "Likert Scale" }
} as const;

export type RatingScale = keyof typeof RATING_SCALES;

// Validation schemas
const RatingValueSchema = z.number()
  .min(0)
  .max(100)
  .brand<"RatingValue">();

const ConfidenceIntervalSchema = z.number()
  .min(0)
  .max(1)
  .brand<"ConfidenceInterval">();

const ReviewCountSchema = z.number()
  .int()
  .min(0)
  .brand<"ReviewCount">();

// Rating metadata
export interface RatingMetadata {
  readonly source: string;
  readonly lastUpdated: Date;
  readonly scale: RatingScale;
  readonly isVerified: boolean;
  readonly sampleSize: ReviewCount;
}

const RatingMetadataSchema = z.object({
  source: z.string().min(1),
  lastUpdated: z.date(),
  scale: z.enum(["FIVE_STAR", "TEN_POINT", "PERCENTAGE", "LIKERT"]),
  isVerified: z.boolean(),
  sampleSize: ReviewCountSchema
});

// Main Rating value object
export interface Rating {
  readonly value: RatingValue;
  readonly confidenceInterval: ConfidenceInterval;
  readonly metadata: RatingMetadata;
  readonly formatted: string;
  readonly createdAt: Date;
}

const RatingSchema = z.object({
  value: RatingValueSchema,
  confidenceInterval: ConfidenceIntervalSchema,
  metadata: RatingMetadataSchema,
  formatted: z.string(),
  createdAt: z.date()
});

// Domain errors
export class InvalidRatingError extends Error {
  constructor(message: string, public readonly field: string) {
    super(message);
    this.name = "InvalidRatingError";
  }
}

export class RatingScaleMismatchError extends Error {
  constructor(scale1: RatingScale, scale2: RatingScale) {
    super(`Cannot perform operation between ${scale1} and ${scale2} scales`);
    this.name = "RatingScaleMismatchError";
  }
}

export class InsufficientDataError extends Error {
  constructor(minSamples: number, actualSamples: number) {
    super(`Insufficient data: need at least ${minSamples} samples, got ${actualSamples}`);
    this.name = "InsufficientDataError";
  }
}

// Factory functions
export const createRatingValue = (value: number, scale: RatingScale): Either<InvalidRatingError, RatingValue> => {
  const scaleConfig = RATING_SCALES[scale];
  const normalizedValue = normalizeToPercentage(value, scale);
  
  try {
    const result = RatingValueSchema.parse(normalizedValue);
    return right(result);
  } catch (error) {
    return left(new InvalidRatingError(
      `Rating value ${value} is invalid for ${scale} scale`,
      "value"
    ));
  }
};

export const createConfidenceInterval = (interval: number): Either<InvalidRatingError, ConfidenceInterval> => {
  try {
    const result = ConfidenceIntervalSchema.parse(interval);
    return right(result);
  } catch (error) {
    return left(new InvalidRatingError("Confidence interval must be between 0 and 1", "confidenceInterval"));
  }
};

export const createReviewCount = (count: number): Either<InvalidRatingError, ReviewCount> => {
  try {
    const result = ReviewCountSchema.parse(count);
    return right(result);
  } catch (error) {
    return left(new InvalidRatingError("Review count must be a non-negative integer", "count"));
  }
};

// Utility function to normalize ratings to percentage scale
function normalizeToPercentage(value: number, scale: RatingScale): number {
  const scaleConfig = RATING_SCALES[scale];
  
  switch (scale) {
    case "FIVE_STAR":
      return ((value - scaleConfig.min) / (scaleConfig.max - scaleConfig.min)) * 100;
    case "TEN_POINT":
      return ((value - scaleConfig.min) / (scaleConfig.max - scaleConfig.min)) * 100;
    case "PERCENTAGE":
      return value;
    case "LIKERT":
      return ((value - scaleConfig.min) / (scaleConfig.max - scaleConfig.min)) * 100;
    default:
      return value;
  }
}

// Utility function to convert percentage back to original scale
function denormalizeFromPercentage(percentage: number, scale: RatingScale): number {
  const scaleConfig = RATING_SCALES[scale];
  
  switch (scale) {
    case "FIVE_STAR":
      return scaleConfig.min + (percentage / 100) * (scaleConfig.max - scaleConfig.min);
    case "TEN_POINT":
      return scaleConfig.min + (percentage / 100) * (scaleConfig.max - scaleConfig.min);
    case "PERCENTAGE":
      return percentage;
    case "LIKERT":
      return Math.round(scaleConfig.min + (percentage / 100) * (scaleConfig.max - scaleConfig.min));
    default:
      return percentage;
  }
}

// Main Rating class with business logic
export class Rating {
  private constructor(
    private readonly _value: RatingValue,
    private readonly _confidenceInterval: ConfidenceInterval,
    private readonly _metadata: RatingMetadata,
    private readonly _createdAt: Date = new Date()
  ) {}

  static create(
    value: number,
    scale: RatingScale,
    confidenceInterval: number = 0.95,
    metadata: Omit<RatingMetadata, "lastUpdated">
  ): Either<InvalidRatingError, Rating> {
    const valueResult = createRatingValue(value, scale);
    if (valueResult._tag === "Left") {
      return left(valueResult.left);
    }

    const confidenceResult = createConfidenceInterval(confidenceInterval);
    if (confidenceResult._tag === "Left") {
      return left(confidenceResult.left);
    }

    const countResult = createReviewCount(metadata.sampleSize);
    if (countResult._tag === "Left") {
      return left(countResult.left);
    }

    const rating = new Rating(
      valueResult.right,
      confidenceResult.right,
      {
        ...metadata,
        sampleSize: countResult.right,
        lastUpdated: new Date()
      }
    );

    return right(rating);
  }

  static fromReviews(
    reviews: number[],
    scale: RatingScale,
    source: string
  ): Either<InsufficientDataError | InvalidRatingError, Rating> {
    if (reviews.length < 3) {
      return left(new InsufficientDataError(3, reviews.length));
    }

    const average = reviews.reduce((sum, review) => sum + review, 0) / reviews.length;
    const variance = reviews.reduce((sum, review) => sum + Math.pow(review - average, 2), 0) / reviews.length;
    const standardError = Math.sqrt(variance / reviews.length);
    const confidenceInterval = Math.min(standardError * 1.96, 0.1); // 95% CI, max 10%

    return Rating.create(
      average,
      scale,
      confidenceInterval,
      {
        source,
        scale,
        isVerified: reviews.length >= 10,
        sampleSize: reviews.length
      }
    );
  }

  // Getters
  get value(): RatingValue { return this._value; }
  get confidenceInterval(): ConfidenceInterval { return this._confidenceInterval; }
  get metadata(): RatingMetadata { return this._metadata; }
  get createdAt(): Date { return this._createdAt; }

  // Formatting
  get formatted(): string {
    const originalValue = denormalizeFromPercentage(this._value, this._metadata.scale);
    const scaleConfig = RATING_SCALES[this._metadata.scale];
    
    switch (this._metadata.scale) {
      case "FIVE_STAR":
        return `${originalValue.toFixed(1)}/5 ⭐`;
      case "TEN_POINT":
        return `${originalValue.toFixed(1)}/10`;
      case "PERCENTAGE":
        return `${originalValue.toFixed(0)}%`;
      case "LIKERT":
        return `${originalValue}/7`;
      default:
        return `${originalValue.toFixed(1)}`;
    }
  }

  get formattedWithConfidence(): string {
    const originalValue = denormalizeFromPercentage(this._value, this._metadata.scale);
    const margin = denormalizeFromPercentage(this._confidenceInterval * 100, this._metadata.scale);
    
    return `${this.formatted} (±${margin.toFixed(1)})`;
  }

  // Statistical operations
  add(other: Rating): Either<RatingScaleMismatchError, Rating> {
    if (this._metadata.scale !== other._metadata.scale) {
      return left(new RatingScaleMismatchError(this._metadata.scale, other._metadata.scale));
    }

    // Weighted average based on sample sizes
    const totalSamples = this._metadata.sampleSize + other._metadata.sampleSize;
    const weightedValue = (
      (this._value * this._metadata.sampleSize) + 
      (other._value * other._metadata.sampleSize)
    ) / totalSamples;

    // Combined confidence interval (simplified)
    const combinedConfidence = Math.sqrt(
      (this._confidenceInterval * this._confidenceInterval + 
       other._confidenceInterval * other._confidenceInterval) / 2
    );

    return Rating.create(
      denormalizeFromPercentage(weightedValue, this._metadata.scale),
      this._metadata.scale,
      combinedConfidence,
      {
        source: `${this._metadata.source} + ${other._metadata.source}`,
        scale: this._metadata.scale,
        isVerified: this._metadata.isVerified && other._metadata.isVerified,
        sampleSize: totalSamples
      }
    );
  }

  // Comparison operations
  equals(other: Rating): boolean {
    return this._value === other._value && 
           this._metadata.scale === other._metadata.scale;
  }

  isGreaterThan(other: Rating): Either<RatingScaleMismatchError, boolean> {
    if (this._metadata.scale !== other._metadata.scale) {
      return left(new RatingScaleMismatchError(this._metadata.scale, other._metadata.scale));
    }
    return right(this._value > other._value);
  }

  isLessThan(other: Rating): Either<RatingScaleMismatchError, boolean> {
    if (this._metadata.scale !== other._metadata.scale) {
      return left(new RatingScaleMismatchError(this._metadata.scale, other._metadata.scale));
    }
    return right(this._value < other._value);
  }

  // Quality assessment
  get quality(): "excellent" | "good" | "fair" | "poor" {
    if (this._value >= 90) return "excellent";
    if (this._value >= 75) return "good";
    if (this._value >= 60) return "fair";
    return "poor";
  }

  get reliability(): "high" | "medium" | "low" {
    if (this._confidenceInterval <= 0.05 && this._metadata.sampleSize >= 20) return "high";
    if (this._confidenceInterval <= 0.1 && this._metadata.sampleSize >= 10) return "medium";
    return "low";
  }

  // Validation
  isValid(): boolean {
    return RatingSchema.safeParse({
      value: this._value,
      confidenceInterval: this._confidenceInterval,
      metadata: this._metadata,
      formatted: this.formatted,
      createdAt: this._createdAt
    }).success;
  }

  // Serialization
  toJSON(): { 
    value: number; 
    confidenceInterval: number; 
    metadata: RatingMetadata; 
    formatted: string; 
    createdAt: string 
  } {
    return {
      value: this._value,
      confidenceInterval: this._confidenceInterval,
      metadata: this._metadata,
      formatted: this.formatted,
      createdAt: this._createdAt.toISOString()
    };
  }

  static fromJSON(data: { 
    value: number; 
    confidenceInterval: number; 
    metadata: RatingMetadata; 
    createdAt: string 
  }): Either<InvalidRatingError, Rating> {
    const valueResult = createRatingValue(data.value, data.metadata.scale);
    if (valueResult._tag === "Left") {
      return left(valueResult.left);
    }

    const confidenceResult = createConfidenceInterval(data.confidenceInterval);
    if (confidenceResult._tag === "Left") {
      return left(confidenceResult.left);
    }

    const rating = new Rating(
      valueResult.right,
      confidenceResult.right,
      data.metadata,
      new Date(data.createdAt)
    );

    return right(rating);
  }
}

// Composite rating for multiple criteria
export interface CompositeRating {
  readonly overall: Rating;
  readonly criteria: Map<string, Rating>;
  readonly weights: Map<string, number>;
  readonly lastUpdated: Date;
}

export class CompositeRating {
  private constructor(
    private readonly _overall: Rating,
    private readonly _criteria: Map<string, Rating>,
    private readonly _weights: Map<string, number>,
    private readonly _lastUpdated: Date = new Date()
  ) {}

  static create(
    criteria: Map<string, Rating>,
    weights: Map<string, number>
  ): Either<InvalidRatingError, CompositeRating> {
    // Validate weights sum to 1
    const totalWeight = Array.from(weights.values()).reduce((sum, weight) => sum + weight, 0);
    if (Math.abs(totalWeight - 1) > 0.001) {
      return left(new InvalidRatingError("Weights must sum to 1", "weights"));
    }

    // Calculate weighted average
    let weightedSum = 0;
    for (const [criterion, rating] of criteria) {
      const weight = weights.get(criterion) || 0;
      weightedSum += rating.value * weight;
    }

    const overallResult = Rating.create(
      denormalizeFromPercentage(weightedSum, criteria.values().next().value.metadata.scale),
      criteria.values().next().value.metadata.scale,
      0.1, // Default confidence interval
      {
        source: "composite",
        scale: criteria.values().next().value.metadata.scale,
        isVerified: Array.from(criteria.values()).every(r => r.metadata.isVerified),
        sampleSize: Math.min(...Array.from(criteria.values()).map(r => r.metadata.sampleSize))
      }
    );

    if (overallResult._tag === "Left") {
      return left(overallResult.left);
    }

    const composite = new CompositeRating(
      overallResult.right,
      criteria,
      weights
    );

    return right(composite);
  }

  // Getters
  get overall(): Rating { return this._overall; }
  get criteria(): Map<string, Rating> { return this._criteria; }
  get weights(): Map<string, number> { return this._weights; }
  get lastUpdated(): Date { return this._lastUpdated; }

  // Analysis
  get topCriteria(): string[] {
    return Array.from(this._criteria.entries())
      .sort(([, a], [, b]) => b.value - a.value)
      .map(([criterion]) => criterion);
  }

  get bottomCriteria(): string[] {
    return Array.from(this._criteria.entries())
      .sort(([, a], [, b]) => a.value - b.value)
      .map(([criterion]) => criterion);
  }

  // Serialization
  toJSON(): { 
    overall: any; 
    criteria: Record<string, any>; 
    weights: Record<string, number>; 
    lastUpdated: string 
  } {
    return {
      overall: this._overall.toJSON(),
      criteria: Object.fromEntries(
        Array.from(this._criteria.entries()).map(([k, v]) => [k, v.toJSON()])
      ),
      weights: Object.fromEntries(this._weights),
      lastUpdated: this._lastUpdated.toISOString()
    };
  }
}

// Utility functions
export const formatRating = (value: number, scale: RatingScale): string => {
  const rating = Rating.create(value, scale, 0.1, {
    source: "formatter",
    scale,
    isVerified: false,
    sampleSize: 1
  });
  
  if (rating._tag === "Right") {
    return rating.right.formatted;
  }
  return `${value}`;
};

export const compareRatings = (a: Rating, b: Rating): number => {
  if (a.metadata.scale !== b.metadata.scale) {
    return 0; // Cannot compare different scales
  }
  return a.value - b.value;
};

export const sortRatings = (ratings: Rating[]): Rating[] => {
  return ratings.sort(compareRatings);
};

export const calculateAverageRating = (ratings: Rating[]): Rating | null => {
  if (ratings.length === 0) return null;
  
  const scale = ratings[0].metadata.scale;
  if (!ratings.every(r => r.metadata.scale === scale)) {
    return null; // Cannot average different scales
  }

  const totalValue = ratings.reduce((sum, rating) => sum + rating.value, 0);
  const averageValue = totalValue / ratings.length;
  const totalSamples = ratings.reduce((sum, rating) => sum + rating.metadata.sampleSize, 0);

  return Rating.create(
    denormalizeFromPercentage(averageValue, scale),
    scale,
    0.1,
    {
      source: "average",
      scale,
      isVerified: ratings.every(r => r.metadata.isVerified),
      sampleSize: totalSamples
    }
  ).getOrElse(null as any);
};
