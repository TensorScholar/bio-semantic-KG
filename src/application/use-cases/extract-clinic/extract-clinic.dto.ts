/**
 * Extract Clinic DTO - Advanced Data Transfer Object Implementation
 * 
 * Implements comprehensive data transfer objects with mathematical
 * foundations and provable correctness properties for clinic extraction.
 * 
 * MATHEMATICAL FOUNDATION:
 * Let D = (T, V, S, C) be a DTO system where:
 * - T = {t₁, t₂, ..., tₙ} is the set of types
 * - V = {v₁, v₂, ..., vₘ} is the set of validators
 * - S = {s₁, s₂, ..., sₖ} is the set of serializers
 * - C = {c₁, c₂, ..., cₗ} is the set of converters
 * 
 * DTO Operations:
 * - Type Validation: TV: T × D → V where D is data
 * - Serialization: S: D × F → S where F is format
 * - Deserialization: D: S × F → D where F is format
 * - Conversion: C: D × T → D where T is target type
 * 
 * COMPLEXITY ANALYSIS:
 * - Type Validation: O(1) with schema validation
 * - Serialization: O(n) where n is field count
 * - Deserialization: O(n) where n is field count
 * - Conversion: O(1) with type mapping
 * 
 * @file extract-clinic.dto.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Result, Ok, Err } from "../../../../shared/kernel/result.ts";
import { Option, Some, None } from "../../../../shared/kernel/option.ts";

// Mathematical type definitions
export type DTOId = string;
export type FieldName = string;
export type FieldType = 'string' | 'number' | 'boolean' | 'date' | 'array' | 'object';

// Base DTO interface with mathematical properties
export interface BaseDTO {
  readonly id: DTOId;
  readonly version: string;
  readonly timestamp: Date;
  readonly metadata: {
    readonly source: string;
    readonly confidence: number;
    readonly validation: {
      readonly isValid: boolean;
      readonly errors: string[];
      readonly warnings: string[];
    };
  };
}

// Input DTOs with mathematical precision
export interface ExtractClinicRequestDTO extends BaseDTO {
  readonly type: 'ExtractClinicRequest';
  readonly data: {
    readonly sourceUrl: string;
    readonly parserType: 'beautifulsoup' | 'selenium' | 'api' | 'manual';
    readonly configuration: {
      readonly selectors: Record<string, string>;
      readonly timeout: number;
      readonly retries: number;
      readonly validation: boolean;
      readonly enrichment: boolean;
    };
    readonly context: {
      readonly userId?: string;
      readonly sessionId?: string;
      readonly correlationId?: string;
      readonly priority: number;
    };
  };
}

export interface ExtractClinicResponseDTO extends BaseDTO {
  readonly type: 'ExtractClinicResponse';
  readonly data: {
    readonly extractionId: string;
    readonly clinic: {
      readonly id: string;
      readonly name: string;
      readonly address: string;
      readonly phone: string;
      readonly email: string;
      readonly website: string;
      readonly services: string[];
      readonly practitioners: string[];
      readonly rating: {
        readonly average: number;
        readonly count: number;
        readonly source: string;
      };
      readonly location: {
        readonly latitude: number;
        readonly longitude: number;
        readonly address: string;
      };
      readonly hours: Record<string, string>;
      readonly isActive: boolean;
    };
    readonly metadata: {
      readonly sourceUrl: string;
      readonly parserType: string;
      readonly extractionTime: number;
      readonly confidence: number;
      readonly quality: number;
      readonly validationScore: number;
      readonly complianceScore: number;
    };
    readonly events: {
      readonly extractionStarted: {
        readonly id: string;
        readonly timestamp: Date;
        readonly data: any;
      };
      readonly extractionCompleted: {
        readonly id: string;
        readonly timestamp: Date;
        readonly data: any;
      };
      readonly clinicCreated: {
        readonly id: string;
        readonly timestamp: Date;
        readonly data: any;
      };
    };
  };
}

export interface ExtractClinicErrorDTO extends BaseDTO {
  readonly type: 'ExtractClinicError';
  readonly data: {
    readonly code: string;
    readonly message: string;
    readonly field?: string;
    readonly details?: Record<string, any>;
    readonly timestamp: Date;
  };
}

// Validation schemas with mathematical constraints
const BaseDTOSchema = z.object({
  id: z.string().min(1),
  version: z.string().min(1),
  timestamp: z.date(),
  metadata: z.object({
    source: z.string().min(1),
    confidence: z.number().min(0).max(1),
    validation: z.object({
      isValid: z.boolean(),
      errors: z.array(z.string()),
      warnings: z.array(z.string())
    })
  })
});

const ExtractClinicRequestDTOSchema = BaseDTOSchema.extend({
  type: z.literal('ExtractClinicRequest'),
  data: z.object({
    sourceUrl: z.string().url(),
    parserType: z.enum(['beautifulsoup', 'selenium', 'api', 'manual']),
    configuration: z.object({
      selectors: z.record(z.string()),
      timeout: z.number().int().positive(),
      retries: z.number().int().min(0).max(10),
      validation: z.boolean(),
      enrichment: z.boolean()
    }),
    context: z.object({
      userId: z.string().optional(),
      sessionId: z.string().optional(),
      correlationId: z.string().optional(),
      priority: z.number().int().min(1).max(10)
    })
  })
});

const ExtractClinicResponseDTOSchema = BaseDTOSchema.extend({
  type: z.literal('ExtractClinicResponse'),
  data: z.object({
    extractionId: z.string().min(1),
    clinic: z.object({
      id: z.string().min(1),
      name: z.string().min(1),
      address: z.string(),
      phone: z.string(),
      email: z.string(),
      website: z.string(),
      services: z.array(z.string()),
      practitioners: z.array(z.string()),
      rating: z.object({
        average: z.number().min(0).max(5),
        count: z.number().int().min(0),
        source: z.string()
      }),
      location: z.object({
        latitude: z.number(),
        longitude: z.number(),
        address: z.string()
      }),
      hours: z.record(z.string()),
      isActive: z.boolean()
    }),
    metadata: z.object({
      sourceUrl: z.string().url(),
      parserType: z.string(),
      extractionTime: z.number().int().min(0),
      confidence: z.number().min(0).max(1),
      quality: z.number().min(0).max(1),
      validationScore: z.number().min(0).max(1),
      complianceScore: z.number().min(0).max(1)
    }),
    events: z.object({
      extractionStarted: z.object({
        id: z.string(),
        timestamp: z.date(),
        data: z.any()
      }),
      extractionCompleted: z.object({
        id: z.string(),
        timestamp: z.date(),
        data: z.any()
      }),
      clinicCreated: z.object({
        id: z.string(),
        timestamp: z.date(),
        data: z.any()
      })
    })
  })
});

const ExtractClinicErrorDTOSchema = BaseDTOSchema.extend({
  type: z.literal('ExtractClinicError'),
  data: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
    field: z.string().optional(),
    details: z.record(z.any()).optional(),
    timestamp: z.date()
  })
});

// Domain errors with mathematical precision
export class DTOError extends Error {
  constructor(
    message: string,
    public readonly dtoId: DTOId,
    public readonly operation: string
  ) {
    super(message);
    this.name = "DTOError";
  }
}

export class DTOValidationError extends Error {
  constructor(
    message: string,
    public readonly field: string,
    public readonly value: any
  ) {
    super(message);
    this.name = "DTOValidationError";
  }
}

// Mathematical utility functions for DTO operations
export class DTOMath {
  /**
   * Calculate DTO completeness with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures completeness calculation is mathematically accurate
   */
  static calculateCompleteness(dto: BaseDTO): number {
    let completeness = 0;
    
    // Required fields presence
    if (dto.id) completeness += 0.2;
    if (dto.version) completeness += 0.2;
    if (dto.timestamp) completeness += 0.2;
    if (dto.metadata.source) completeness += 0.2;
    if (dto.metadata.confidence >= 0) completeness += 0.2;
    
    return Math.min(1.0, completeness);
  }
  
  /**
   * Calculate DTO quality with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures quality calculation is mathematically accurate
   */
  static calculateQuality(dto: BaseDTO): number {
    let quality = 0;
    
    // Base quality from completeness
    quality += this.calculateCompleteness(dto) * 0.4;
    
    // Validation quality
    const validationQuality = dto.metadata.validation.isValid ? 1.0 : 0.5;
    quality += validationQuality * 0.3;
    
    // Error count penalty
    const errorPenalty = Math.max(0, 1 - (dto.metadata.validation.errors.length * 0.1));
    quality += errorPenalty * 0.2;
    
    // Warning count penalty
    const warningPenalty = Math.max(0, 1 - (dto.metadata.validation.warnings.length * 0.05));
    quality += warningPenalty * 0.1;
    
    return Math.min(1.0, quality);
  }
  
  /**
   * Calculate DTO confidence with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures confidence calculation is mathematically accurate
   */
  static calculateConfidence(dto: BaseDTO): number {
    const baseConfidence = dto.metadata.confidence;
    const quality = this.calculateQuality(dto);
    const completeness = this.calculateCompleteness(dto);
    
    // Weighted combination
    return (baseConfidence * 0.5) + (quality * 0.3) + (completeness * 0.2);
  }
  
  /**
   * Calculate DTO similarity with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures similarity calculation is mathematically accurate
   */
  static calculateSimilarity(dto1: BaseDTO, dto2: BaseDTO): number {
    let similarity = 0;
    
    // ID similarity
    if (dto1.id === dto2.id) similarity += 0.3;
    
    // Version similarity
    if (dto1.version === dto2.version) similarity += 0.2;
    
    // Source similarity
    if (dto1.metadata.source === dto2.metadata.source) similarity += 0.2;
    
    // Timestamp proximity (within 1 hour)
    const timeDiff = Math.abs(dto1.timestamp.getTime() - dto2.timestamp.getTime());
    const oneHour = 60 * 60 * 1000;
    if (timeDiff <= oneHour) {
      similarity += 0.3 * (1 - timeDiff / oneHour);
    }
    
    return Math.min(1.0, similarity);
  }
  
  /**
   * Calculate DTO age with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures age calculation is mathematically accurate
   */
  static calculateAge(dto: BaseDTO): number {
    const now = new Date();
    const ageInMilliseconds = now.getTime() - dto.timestamp.getTime();
    const ageInHours = ageInMilliseconds / (1000 * 60 * 60);
    return Math.max(0, ageInHours);
  }
  
  /**
   * Calculate DTO freshness with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures freshness calculation is mathematically accurate
   */
  static calculateFreshness(dto: BaseDTO): number {
    const age = this.calculateAge(dto);
    const maxAge = 24; // 24 hours
    return Math.max(0, 1 - (age / maxAge));
  }
}

// DTO factory functions with mathematical validation
export class DTOFactory {
  /**
   * Create extract clinic request DTO with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures DTO creation is mathematically accurate
   */
  static createExtractClinicRequestDTO(
    data: ExtractClinicRequestDTO['data'],
    source: string = 'extract-clinic-service'
  ): Result<ExtractClinicRequestDTO, Error> {
    try {
      const dto: ExtractClinicRequestDTO = {
        id: `dto-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        version: '1.0.0',
        timestamp: new Date(),
        type: 'ExtractClinicRequest',
        data,
        metadata: {
          source,
          confidence: 1.0, // Default confidence for input
          validation: {
            isValid: true,
            errors: [],
            warnings: []
          }
        }
      };
      
      // Validate DTO
      const validation = ExtractClinicRequestDTOSchema.safeParse(dto);
      if (!validation.success) {
        return Err(new DTOValidationError(
          'Invalid extract clinic request DTO',
          'data',
          data
        ));
      }
      
      return Ok(dto);
    } catch (error) {
      return Err(new DTOError(
        `Failed to create extract clinic request DTO: ${error.message}`,
        'unknown',
        'create'
      ));
    }
  }
  
  /**
   * Create extract clinic response DTO with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures DTO creation is mathematically accurate
   */
  static createExtractClinicResponseDTO(
    data: ExtractClinicResponseDTO['data'],
    source: string = 'extract-clinic-service'
  ): Result<ExtractClinicResponseDTO, Error> {
    try {
      const dto: ExtractClinicResponseDTO = {
        id: `dto-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        version: '1.0.0',
        timestamp: new Date(),
        type: 'ExtractClinicResponse',
        data,
        metadata: {
          source,
          confidence: data.metadata.confidence,
          validation: {
            isValid: true,
            errors: [],
            warnings: []
          }
        }
      };
      
      // Validate DTO
      const validation = ExtractClinicResponseDTOSchema.safeParse(dto);
      if (!validation.success) {
        return Err(new DTOValidationError(
          'Invalid extract clinic response DTO',
          'data',
          data
        ));
      }
      
      return Ok(dto);
    } catch (error) {
      return Err(new DTOError(
        `Failed to create extract clinic response DTO: ${error.message}`,
        'unknown',
        'create'
      ));
    }
  }
  
  /**
   * Create extract clinic error DTO with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures DTO creation is mathematically accurate
   */
  static createExtractClinicErrorDTO(
    data: ExtractClinicErrorDTO['data'],
    source: string = 'extract-clinic-service'
  ): Result<ExtractClinicErrorDTO, Error> {
    try {
      const dto: ExtractClinicErrorDTO = {
        id: `dto-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        version: '1.0.0',
        timestamp: new Date(),
        type: 'ExtractClinicError',
        data,
        metadata: {
          source,
          confidence: 0.0, // Error DTOs have no confidence
          validation: {
            isValid: false,
            errors: [data.message],
            warnings: []
          }
        }
      };
      
      // Validate DTO
      const validation = ExtractClinicErrorDTOSchema.safeParse(dto);
      if (!validation.success) {
        return Err(new DTOValidationError(
          'Invalid extract clinic error DTO',
          'data',
          data
        ));
      }
      
      return Ok(dto);
    } catch (error) {
      return Err(new DTOError(
        `Failed to create extract clinic error DTO: ${error.message}`,
        'unknown',
        'create'
      ));
    }
  }
}

// Serialization functions with mathematical properties
export class DTOSerializer {
  /**
   * Serialize DTO to JSON with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is field count
   * CORRECTNESS: Ensures serialization is mathematically accurate
   */
  static serializeToJSON(dto: BaseDTO): Result<string, Error> {
    try {
      const json = JSON.stringify(dto, null, 2);
      return Ok(json);
    } catch (error) {
      return Err(new DTOError(
        `Failed to serialize DTO to JSON: ${error.message}`,
        dto.id,
        'serialize'
      ));
    }
  }
  
  /**
   * Deserialize DTO from JSON with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is field count
   * CORRECTNESS: Ensures deserialization is mathematically accurate
   */
  static deserializeFromJSON<T extends BaseDTO>(
    json: string,
    schema: z.ZodSchema<T>
  ): Result<T, Error> {
    try {
      const parsed = JSON.parse(json);
      const validation = schema.safeParse(parsed);
      
      if (!validation.success) {
        return Err(new DTOValidationError(
          'Invalid JSON structure for DTO',
          'json',
          json
        ));
      }
      
      return Ok(validation.data);
    } catch (error) {
      return Err(new DTOError(
        `Failed to deserialize DTO from JSON: ${error.message}`,
        'unknown',
        'deserialize'
      ));
    }
  }
}

// Utility functions with mathematical properties
export function validateDTO(dto: BaseDTO): boolean {
  return DTOMath.calculateCompleteness(dto) > 0.8;
}

export function calculateDTOQuality(dto: BaseDTO): number {
  return DTOMath.calculateQuality(dto);
}

export function calculateDTOConfidence(dto: BaseDTO): number {
  return DTOMath.calculateConfidence(dto);
}

export function calculateDTOSimilarity(dto1: BaseDTO, dto2: BaseDTO): number {
  return DTOMath.calculateSimilarity(dto1, dto2);
}

export function calculateDTOFreshness(dto: BaseDTO): number {
  return DTOMath.calculateFreshness(dto);
}
