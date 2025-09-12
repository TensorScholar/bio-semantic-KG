/**
 * Price Range Value Object - Advanced Pricing Management
 * 
 * Implements comprehensive price range domain with mathematical
 * foundations and provable correctness properties for service pricing.
 * 
 * MATHEMATICAL FOUNDATION:
 * Let P = (L, H, C, M) be a price range system where:
 * - L = {l₁, l₂, ..., lₙ} is the set of lower bounds
 * - H = {h₁, h₂, ..., hₘ} is the set of upper bounds
 * - C = {c₁, c₂, ..., cₖ} is the set of currencies
 * - M = {m₁, m₂, ..., mₗ} is the set of market conditions
 * 
 * Price Range Operations:
 * - Range Validation: RV: L × H → V where V is validation
 * - Currency Conversion: CC: P × R → P where R is rate
 * - Market Adjustment: MA: P × M → P where M is market
 * - Price Interpolation: PI: P × F → P where F is factor
 * 
 * COMPLEXITY ANALYSIS:
 * - Range Validation: O(1) with boundary checks
 * - Currency Conversion: O(1) with rate lookup
 * - Market Adjustment: O(1) with factor application
 * - Price Interpolation: O(1) with linear interpolation
 * 
 * @file price-range.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Result, Ok, Err } from "../../../shared/kernel/result.ts";
import { Option, Some, None } from "../../../shared/kernel/option.ts";

// Mathematical type definitions
export type CurrencyCode = string;
export type PriceValue = number;
export type MarketCondition = 'low' | 'medium' | 'high' | 'premium';

// Price range entities with mathematical properties
export interface PriceRange {
  readonly min: PriceValue;
  readonly max: PriceValue;
  readonly currency: CurrencyCode;
  readonly marketCondition: MarketCondition;
  readonly confidence: number; // 0-1 scale
  readonly metadata: {
    readonly source: string;
    readonly lastUpdated: Date;
    readonly sampleSize: number;
    readonly standardDeviation: number;
    readonly median: number;
    readonly mode: number;
  };
}

// Validation schemas with mathematical constraints
const CurrencyCodeSchema = z.string()
  .length(3)
  .regex(/^[A-Z]{3}$/, "Currency code must be 3 uppercase letters");

const PriceValueSchema = z.number()
  .positive()
  .finite()
  .safe();

const MarketConditionSchema = z.enum(['low', 'medium', 'high', 'premium']);

const PriceRangeSchema = z.object({
  min: PriceValueSchema,
  max: PriceValueSchema,
  currency: CurrencyCodeSchema,
  marketCondition: MarketConditionSchema,
  confidence: z.number().min(0).max(1),
  metadata: z.object({
    source: z.string().min(1),
    lastUpdated: z.date(),
    sampleSize: z.number().int().min(0),
    standardDeviation: z.number().min(0),
    median: z.number().positive(),
    mode: z.number().positive()
  })
});

// Domain errors with mathematical precision
export class PriceRangeError extends Error {
  constructor(
    message: string,
    public readonly operation: string
  ) {
    super(message);
    this.name = "PriceRangeError";
  }
}

export class PriceValidationError extends Error {
  constructor(
    message: string,
    public readonly field: string,
    public readonly value: number
  ) {
    super(message);
    this.name = "PriceValidationError";
  }
}

// Mathematical utility functions for price range operations
export class PriceRangeMath {
  /**
   * Calculate price range validity with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures validity calculation is mathematically accurate
   */
  static calculateValidity(priceRange: PriceRange): Result<boolean, Error> {
    try {
      // Basic range validation
      if (priceRange.min >= priceRange.max) {
        return Err(new PriceValidationError(
          "Minimum price must be less than maximum price",
          'min',
          priceRange.min
        ));
      }
      
      // Range width validation (too narrow or too wide)
      const rangeWidth = priceRange.max - priceRange.min;
      const minWidth = priceRange.min * 0.1; // 10% of minimum price
      const maxWidth = priceRange.min * 10; // 10x minimum price
      
      if (rangeWidth < minWidth) {
        return Err(new PriceValidationError(
          "Price range is too narrow",
          'range',
          rangeWidth
        ));
      }
      
      if (rangeWidth > maxWidth) {
        return Err(new PriceValidationError(
          "Price range is too wide",
          'range',
          rangeWidth
        ));
      }
      
      // Confidence validation
      if (priceRange.confidence < 0 || priceRange.confidence > 1) {
        return Err(new PriceValidationError(
          "Confidence must be between 0 and 1",
          'confidence',
          priceRange.confidence
        ));
      }
      
      return Ok(true);
    } catch (error) {
      return Err(new PriceRangeError(
        `Price range validation failed: ${error.message}`,
        'validation'
      ));
    }
  }
  
  /**
   * Calculate price range center with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures center calculation is mathematically accurate
   */
  static calculateCenter(priceRange: PriceRange): number {
    return (priceRange.min + priceRange.max) / 2;
  }
  
  /**
   * Calculate price range width with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures width calculation is mathematically accurate
   */
  static calculateWidth(priceRange: PriceRange): number {
    return priceRange.max - priceRange.min;
  }
  
  /**
   * Calculate price range variance with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures variance calculation is mathematically accurate
   */
  static calculateVariance(priceRange: PriceRange): number {
    const center = this.calculateCenter(priceRange);
    const width = this.calculateWidth(priceRange);
    return Math.pow(width / 2, 2) / 3; // Uniform distribution variance
  }
  
  /**
   * Calculate price range standard deviation with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures standard deviation calculation is mathematically accurate
   */
  static calculateStandardDeviation(priceRange: PriceRange): number {
    return Math.sqrt(this.calculateVariance(priceRange));
  }
  
  /**
   * Calculate price range coefficient of variation with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures coefficient calculation is mathematically accurate
   */
  static calculateCoefficientOfVariation(priceRange: PriceRange): number {
    const center = this.calculateCenter(priceRange);
    const stdDev = this.calculateStandardDeviation(priceRange);
    return center > 0 ? stdDev / center : 0;
  }
  
  /**
   * Calculate market adjustment factor with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures adjustment calculation is mathematically accurate
   */
  static calculateMarketAdjustmentFactor(marketCondition: MarketCondition): number {
    const factors: Record<MarketCondition, number> = {
      'low': 0.8,
      'medium': 1.0,
      'high': 1.2,
      'premium': 1.5
    };
    
    return factors[marketCondition];
  }
  
  /**
   * Apply market adjustment with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures adjustment application is mathematically accurate
   */
  static applyMarketAdjustment(priceRange: PriceRange, newMarketCondition: MarketCondition): PriceRange {
    const currentFactor = this.calculateMarketAdjustmentFactor(priceRange.marketCondition);
    const newFactor = this.calculateMarketAdjustmentFactor(newMarketCondition);
    const adjustmentRatio = newFactor / currentFactor;
    
    return {
      ...priceRange,
      min: priceRange.min * adjustmentRatio,
      max: priceRange.max * adjustmentRatio,
      marketCondition: newMarketCondition,
      metadata: {
        ...priceRange.metadata,
        lastUpdated: new Date()
      }
    };
  }
  
  /**
   * Calculate currency conversion with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures conversion calculation is mathematically accurate
   */
  static convertCurrency(
    priceRange: PriceRange,
    targetCurrency: CurrencyCode,
    exchangeRate: number
  ): PriceRange {
    if (exchangeRate <= 0) {
      throw new PriceRangeError("Exchange rate must be positive", 'convert_currency');
    }
    
    return {
      ...priceRange,
      min: priceRange.min * exchangeRate,
      max: priceRange.max * exchangeRate,
      currency: targetCurrency,
      metadata: {
        ...priceRange.metadata,
        lastUpdated: new Date()
      }
    };
  }
  
  /**
   * Calculate price interpolation with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures interpolation calculation is mathematically accurate
   */
  static interpolatePrice(priceRange: PriceRange, factor: number): number {
    if (factor < 0 || factor > 1) {
      throw new PriceRangeError("Interpolation factor must be between 0 and 1", 'interpolate');
    }
    
    return priceRange.min + (priceRange.max - priceRange.min) * factor;
  }
  
  /**
   * Calculate price range overlap with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures overlap calculation is mathematically accurate
   */
  static calculateOverlap(range1: PriceRange, range2: PriceRange): number {
    const overlapMin = Math.max(range1.min, range2.min);
    const overlapMax = Math.min(range1.max, range2.max);
    
    if (overlapMin >= overlapMax) return 0;
    
    const overlapWidth = overlapMax - overlapMin;
    const totalWidth = Math.max(range1.max, range2.max) - Math.min(range1.min, range2.min);
    
    return totalWidth > 0 ? overlapWidth / totalWidth : 0;
  }
  
  /**
   * Calculate price range similarity with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures similarity calculation is mathematically accurate
   */
  static calculateSimilarity(range1: PriceRange, range2: PriceRange): number {
    // Currency must match
    if (range1.currency !== range2.currency) return 0;
    
    // Calculate center similarity
    const center1 = this.calculateCenter(range1);
    const center2 = this.calculateCenter(range2);
    const centerSimilarity = 1 - Math.abs(center1 - center2) / Math.max(center1, center2);
    
    // Calculate width similarity
    const width1 = this.calculateWidth(range1);
    const width2 = this.calculateWidth(range2);
    const widthSimilarity = 1 - Math.abs(width1 - width2) / Math.max(width1, width2);
    
    // Calculate overlap
    const overlap = this.calculateOverlap(range1, range2);
    
    // Weighted combination
    return (centerSimilarity * 0.4) + (widthSimilarity * 0.3) + (overlap * 0.3);
  }
  
  /**
   * Calculate price range confidence with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures confidence calculation is mathematically accurate
   */
  static calculateConfidence(priceRange: PriceRange): number {
    let confidence = priceRange.confidence;
    
    // Sample size factor
    const sampleSizeFactor = Math.min(1.0, priceRange.metadata.sampleSize / 100);
    confidence *= sampleSizeFactor;
    
    // Standard deviation factor (lower deviation = higher confidence)
    const stdDevFactor = Math.max(0.1, 1 - (priceRange.metadata.standardDeviation / priceRange.min));
    confidence *= stdDevFactor;
    
    // Range width factor (moderate width = higher confidence)
    const width = this.calculateWidth(priceRange);
    const optimalWidth = priceRange.min * 0.5; // 50% of minimum price
    const widthFactor = Math.max(0.1, 1 - Math.abs(width - optimalWidth) / optimalWidth);
    confidence *= widthFactor;
    
    return Math.min(1.0, Math.max(0, confidence));
  }
  
  /**
   * Calculate price range quality score with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures quality calculation is mathematically accurate
   */
  static calculateQualityScore(priceRange: PriceRange): number {
    const validity = this.calculateValidity(priceRange);
    if (validity._tag === "Left") return 0;
    
    const confidence = this.calculateConfidence(priceRange);
    const coefficientOfVariation = this.calculateCoefficientOfVariation(priceRange);
    
    // Lower coefficient of variation = higher quality
    const variationScore = Math.max(0, 1 - coefficientOfVariation);
    
    return (confidence + variationScore) / 2;
  }
}

// Main Price Range Value Object with formal specifications
export class PriceRangeVO {
  private constructor(private readonly data: PriceRange) {}
  
  /**
   * Create price range value object with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures price range is properly created
   */
  static create(data: PriceRange): Result<PriceRangeVO, Error> {
    try {
      // Validate schema
      const validationResult = PriceRangeSchema.safeParse(data);
      if (!validationResult.success) {
        return Err(new PriceRangeError(
          "Invalid price range data",
          "create"
        ));
      }
      
      // Validate price range logic
      const validityResult = PriceRangeMath.calculateValidity(data);
      if (validityResult._tag === "Left") {
        return Err(new PriceRangeError(
          `Invalid price range: ${validityResult.left.message}`,
          "create"
        ));
      }
      
      return Ok(new PriceRangeVO(data));
    } catch (error) {
      return Err(new PriceRangeError(
        `Failed to create price range: ${error.message}`,
        "create"
      ));
    }
  }
  
  /**
   * Get price range data
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures data is properly retrieved
   */
  getData(): PriceRange {
    return this.data;
  }
  
  /**
   * Get minimum price
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures minimum price is properly retrieved
   */
  getMin(): PriceValue {
    return this.data.min;
  }
  
  /**
   * Get maximum price
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures maximum price is properly retrieved
   */
  getMax(): PriceValue {
    return this.data.max;
  }
  
  /**
   * Get currency
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures currency is properly retrieved
   */
  getCurrency(): CurrencyCode {
    return this.data.currency;
  }
  
  /**
   * Get market condition
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures market condition is properly retrieved
   */
  getMarketCondition(): MarketCondition {
    return this.data.marketCondition;
  }
  
  /**
   * Calculate center price
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures center calculation is mathematically accurate
   */
  getCenter(): number {
    return PriceRangeMath.calculateCenter(this.data);
  }
  
  /**
   * Calculate width
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures width calculation is mathematically accurate
   */
  getWidth(): number {
    return PriceRangeMath.calculateWidth(this.data);
  }
  
  /**
   * Calculate variance
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures variance calculation is mathematically accurate
   */
  getVariance(): number {
    return PriceRangeMath.calculateVariance(this.data);
  }
  
  /**
   * Calculate standard deviation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures standard deviation calculation is mathematically accurate
   */
  getStandardDeviation(): number {
    return PriceRangeMath.calculateStandardDeviation(this.data);
  }
  
  /**
   * Calculate coefficient of variation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures coefficient calculation is mathematically accurate
   */
  getCoefficientOfVariation(): number {
    return PriceRangeMath.calculateCoefficientOfVariation(this.data);
  }
  
  /**
   * Calculate confidence score
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures confidence calculation is mathematically accurate
   */
  getConfidence(): number {
    return PriceRangeMath.calculateConfidence(this.data);
  }
  
  /**
   * Calculate quality score
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures quality calculation is mathematically accurate
   */
  getQualityScore(): number {
    return PriceRangeMath.calculateQualityScore(this.data);
  }
  
  /**
   * Interpolate price at factor
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures interpolation calculation is mathematically accurate
   */
  interpolate(factor: number): number {
    return PriceRangeMath.interpolatePrice(this.data, factor);
  }
  
  /**
   * Apply market adjustment
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures adjustment application is mathematically accurate
   */
  applyMarketAdjustment(newMarketCondition: MarketCondition): PriceRangeVO {
    const adjustedRange = PriceRangeMath.applyMarketAdjustment(this.data, newMarketCondition);
    const result = PriceRangeVO.create(adjustedRange);
    if (result._tag === "Left") {
      throw new PriceRangeError(
        `Failed to apply market adjustment: ${result.left.message}`,
        "apply_market_adjustment"
      );
    }
    return result.right;
  }
  
  /**
   * Convert currency
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures conversion calculation is mathematically accurate
   */
  convertCurrency(targetCurrency: CurrencyCode, exchangeRate: number): PriceRangeVO {
    const convertedRange = PriceRangeMath.convertCurrency(this.data, targetCurrency, exchangeRate);
    const result = PriceRangeVO.create(convertedRange);
    if (result._tag === "Left") {
      throw new PriceRangeError(
        `Failed to convert currency: ${result.left.message}`,
        "convert_currency"
      );
    }
    return result.right;
  }
  
  /**
   * Calculate overlap with another price range
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures overlap calculation is mathematically accurate
   */
  calculateOverlap(other: PriceRangeVO): number {
    return PriceRangeMath.calculateOverlap(this.data, other.data);
  }
  
  /**
   * Calculate similarity with another price range
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures similarity calculation is mathematically accurate
   */
  calculateSimilarity(other: PriceRangeVO): number {
    return PriceRangeMath.calculateSimilarity(this.data, other.data);
  }
  
  /**
   * Check if price is within range
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures range check is mathematically accurate
   */
  contains(price: PriceValue): boolean {
    return price >= this.data.min && price <= this.data.max;
  }
  
  /**
   * Check equality with another price range
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures equality check is correct
   */
  equals(other: PriceRangeVO): boolean {
    return this.data.min === other.data.min &&
           this.data.max === other.data.max &&
           this.data.currency === other.data.currency &&
           this.data.marketCondition === other.data.marketCondition;
  }
  
  /**
   * Convert to string representation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures string conversion is correct
   */
  toString(): string {
    return `${this.data.currency} ${this.data.min}-${this.data.max} (${this.data.marketCondition})`;
  }
}

// Factory functions with mathematical validation
export function createPriceRange(data: PriceRange): Result<PriceRangeVO, Error> {
  return PriceRangeVO.create(data);
}

export function validatePriceRange(data: PriceRange): boolean {
  return PriceRangeSchema.safeParse(data).success;
}

export function calculateCenter(priceRange: PriceRange): number {
  return PriceRangeMath.calculateCenter(priceRange);
}

export function calculateWidth(priceRange: PriceRange): number {
  return PriceRangeMath.calculateWidth(priceRange);
}

export function calculateConfidence(priceRange: PriceRange): number {
  return PriceRangeMath.calculateConfidence(priceRange);
}

export function calculateQualityScore(priceRange: PriceRange): number {
  return PriceRangeMath.calculateQualityScore(priceRange);
}
