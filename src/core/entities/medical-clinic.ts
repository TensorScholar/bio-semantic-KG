/**
 * Medical Clinic Entity - Core Domain Model
 * 
 * Represents a comprehensive medical aesthetics clinic with all associated
 * services, practitioners, and operational data. Implements DDD aggregate
 * root pattern with invariant enforcement.
 * 
 * @file medical-clinic.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Branded } from "../../shared/types/branded.ts";
import { Price } from "../value-objects/price.ts";
import { Rating } from "../value-objects/rating.ts";
import { URL } from "../value-objects/url.ts";
import { MedicalProcedure } from "./medical-procedure.ts";
import { Practitioner } from "./practitioner.ts";
import { PatientOutcome } from "./patient-outcome.ts";

// Branded types for type safety
export type ClinicId = Branded<string, "ClinicId">;
export type ClinicName = Branded<string, "ClinicName">;
export type LicenseNumber = Branded<string, "LicenseNumber">;
export type PhoneNumber = Branded<string, "PhoneNumber">;
export type EmailAddress = Branded<string, "EmailAddress">;

// Validation schemas using Zod for runtime type safety
const ClinicIdSchema = z.string().min(1).brand<"ClinicId">();
const ClinicNameSchema = z.string().min(2).max(200).brand<"ClinicName">();
const LicenseNumberSchema = z.string().regex(/^[A-Z0-9-]+$/).brand<"LicenseNumber">();
const PhoneNumberSchema = z.string().regex(/^\+?[\d\s-()]+$/).brand<"PhoneNumber">();
const EmailAddressSchema = z.string().email().brand<"EmailAddress">();

// Address value object
export interface Address {
  readonly street: string;
  readonly city: string;
  readonly state: string;
  readonly postalCode: string;
  readonly country: string;
  readonly coordinates?: {
    readonly latitude: number;
    readonly longitude: number;
  };
}

const AddressSchema = z.object({
  street: z.string().min(1),
  city: z.string().min(1),
  state: z.string().min(1),
  postalCode: z.string().min(1),
  country: z.string().min(1),
  coordinates: z.object({
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180)
  }).optional()
});

// Operating hours
export interface OperatingHours {
  readonly dayOfWeek: number; // 0-6 (Sunday-Saturday)
  readonly openTime: string; // HH:MM format
  readonly closeTime: string; // HH:MM format
  readonly isClosed: boolean;
}

const OperatingHoursSchema = z.object({
  dayOfWeek: z.number().min(0).max(6),
  openTime: z.string().regex(/^([01]?[0-9]|2[0-3]):[0-5][0-9]$/),
  closeTime: z.string().regex(/^([01]?[0-9]|2[0-3]):[0-5][0-9]$/),
  isClosed: z.boolean()
});

// Social media presence
export interface SocialMedia {
  readonly platform: "instagram" | "facebook" | "twitter" | "linkedin" | "youtube";
  readonly handle: string;
  readonly url: URL;
  readonly followerCount?: number;
  readonly engagementRate?: number;
}

const SocialMediaSchema = z.object({
  platform: z.enum(["instagram", "facebook", "twitter", "linkedin", "youtube"]),
  handle: z.string().min(1),
  url: z.string().url(),
  followerCount: z.number().positive().optional(),
  engagementRate: z.number().min(0).max(1).optional()
});

// Certification and accreditation
export interface Certification {
  readonly name: string;
  readonly issuingBody: string;
  readonly issueDate: Date;
  readonly expiryDate?: Date;
  readonly credentialId: string;
  readonly verificationUrl?: URL;
}

const CertificationSchema = z.object({
  name: z.string().min(1),
  issuingBody: z.string().min(1),
  issueDate: z.date(),
  expiryDate: z.date().optional(),
  credentialId: z.string().min(1),
  verificationUrl: z.string().url().optional()
});

// Main MedicalClinic entity
export interface MedicalClinic {
  readonly id: ClinicId;
  readonly name: ClinicName;
  readonly licenseNumber: LicenseNumber;
  readonly address: Address;
  readonly phone: PhoneNumber;
  readonly email: EmailAddress;
  readonly website: URL;
  readonly operatingHours: readonly OperatingHours[];
  readonly socialMedia: readonly SocialMedia[];
  readonly certifications: readonly Certification[];
  readonly services: readonly MedicalProcedure[];
  readonly practitioners: readonly Practitioner[];
  readonly patientOutcomes: readonly PatientOutcome[];
  readonly overallRating: Rating;
  readonly totalReviews: number;
  readonly establishedDate: Date;
  readonly lastUpdated: Date;
  readonly extractionMetadata: {
    readonly sourceUrl: URL;
    readonly extractedAt: Date;
    readonly extractionVersion: string;
    readonly confidenceScore: number;
  };
}

const MedicalClinicSchema = z.object({
  id: ClinicIdSchema,
  name: ClinicNameSchema,
  licenseNumber: LicenseNumberSchema,
  address: AddressSchema,
  phone: PhoneNumberSchema,
  email: EmailAddressSchema,
  website: z.string().url(),
  operatingHours: z.array(OperatingHoursSchema),
  socialMedia: z.array(SocialMediaSchema),
  certifications: z.array(CertificationSchema),
  services: z.array(z.any()), // Will be validated by MedicalProcedure schema
  practitioners: z.array(z.any()), // Will be validated by Practitioner schema
  patientOutcomes: z.array(z.any()), // Will be validated by PatientOutcome schema
  overallRating: z.any(), // Will be validated by Rating schema
  totalReviews: z.number().int().min(0),
  establishedDate: z.date(),
  lastUpdated: z.date(),
  extractionMetadata: z.object({
    sourceUrl: z.string().url(),
    extractedAt: z.date(),
    extractionVersion: z.string(),
    confidenceScore: z.number().min(0).max(1)
  })
});

// Domain errors
export class ClinicValidationError extends Error {
  constructor(message: string, public readonly field: string) {
    super(message);
    this.name = "ClinicValidationError";
  }
}

export class DuplicateServiceError extends Error {
  constructor(serviceName: string) {
    super(`Service '${serviceName}' already exists in clinic`);
    this.name = "DuplicateServiceError";
  }
}

export class InvalidOperatingHoursError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidOperatingHoursError";
  }
}

// Factory functions for creating branded types
export const createClinicId = (id: string): Either<ClinicValidationError, ClinicId> => {
  try {
    const result = ClinicIdSchema.parse(id);
    return right(result);
  } catch (error) {
    return left(new ClinicValidationError("Invalid clinic ID format", "id"));
  }
};

export const createClinicName = (name: string): Either<ClinicValidationError, ClinicName> => {
  try {
    const result = ClinicNameSchema.parse(name);
    return right(result);
  } catch (error) {
    return left(new ClinicValidationError("Invalid clinic name format", "name"));
  }
};

export const createLicenseNumber = (license: string): Either<ClinicValidationError, LicenseNumber> => {
  try {
    const result = LicenseNumberSchema.parse(license);
    return right(result);
  } catch (error) {
    return left(new ClinicValidationError("Invalid license number format", "licenseNumber"));
  }
};

export const createPhoneNumber = (phone: string): Either<ClinicValidationError, PhoneNumber> => {
  try {
    const result = PhoneNumberSchema.parse(phone);
    return right(result);
  } catch (error) {
    return left(new ClinicValidationError("Invalid phone number format", "phone"));
  }
};

export const createEmailAddress = (email: string): Either<ClinicValidationError, EmailAddress> => {
  try {
    const result = EmailAddressSchema.parse(email);
    return right(result);
  } catch (error) {
    return left(new ClinicValidationError("Invalid email address format", "email"));
  }
};

// Business logic methods
export class MedicalClinicAggregate {
  private constructor(private readonly clinic: MedicalClinic) {}

  static create(clinic: MedicalClinic): Either<ClinicValidationError, MedicalClinicAggregate> {
    try {
      MedicalClinicSchema.parse(clinic);
      return right(new MedicalClinicAggregate(clinic));
    } catch (error) {
      if (error instanceof z.ZodError) {
        const firstError = error.errors[0];
        return left(new ClinicValidationError(
          firstError.message,
          firstError.path.join(".")
        ));
      }
      return left(new ClinicValidationError("Unknown validation error", "clinic"));
    }
  }

  // Business invariants
  validateOperatingHours(): Either<InvalidOperatingHoursError, void> {
    for (const hours of this.clinic.operatingHours) {
      if (!hours.isClosed) {
        const openTime = new Date(`2000-01-01T${hours.openTime}:00`);
        const closeTime = new Date(`2000-01-01T${hours.closeTime}:00`);
        
        if (closeTime <= openTime) {
          return left(new InvalidOperatingHoursError(
            `Close time must be after open time for day ${hours.dayOfWeek}`
          ));
        }
      }
    }
    return right(undefined);
  }

  // Service management
  addService(service: MedicalProcedure): Either<DuplicateServiceError, MedicalClinicAggregate> {
    const existingService = this.clinic.services.find(s => s.name === service.name);
    if (existingService) {
      return left(new DuplicateServiceError(service.name));
    }

    const updatedClinic: MedicalClinic = {
      ...this.clinic,
      services: [...this.clinic.services, service],
      lastUpdated: new Date()
    };

    return right(new MedicalClinicAggregate(updatedClinic));
  }

  // Practitioner management
  addPractitioner(practitioner: Practitioner): MedicalClinicAggregate {
    const updatedClinic: MedicalClinic = {
      ...this.clinic,
      practitioners: [...this.clinic.practitioners, practitioner],
      lastUpdated: new Date()
    };

    return new MedicalClinicAggregate(updatedClinic);
  }

  // Analytics and insights
  getServiceCategories(): string[] {
    return [...new Set(this.clinic.services.map(s => s.category))];
  }

  getAverageServicePrice(): Price | null {
    if (this.clinic.services.length === 0) return null;
    
    const totalPrice = this.clinic.services.reduce((sum, service) => {
      return sum + (service.priceRange.min + service.priceRange.max) / 2;
    }, 0);
    
    const averagePrice = totalPrice / this.clinic.services.length;
    return Price.create(averagePrice, this.clinic.services[0].priceRange.currency).getOrElse(
      Price.create(0, "USD").getOrElse(null as any)
    );
  }

  getPractitionerSpecializations(): string[] {
    return [...new Set(this.clinic.practitioners.flatMap(p => p.specializations))];
  }

  // Data completeness scoring
  calculateDataCompleteness(): number {
    const fields = [
      this.clinic.name,
      this.clinic.licenseNumber,
      this.clinic.address,
      this.clinic.phone,
      this.clinic.email,
      this.clinic.website,
      this.clinic.operatingHours.length > 0,
      this.clinic.services.length > 0,
      this.clinic.practitioners.length > 0
    ];

    const completedFields = fields.filter(field => 
      field !== null && field !== undefined && field !== ""
    ).length;

    return completedFields / fields.length;
  }

  // Export immutable data
  toJSON(): MedicalClinic {
    return { ...this.clinic };
  }

  // Getters for read-only access
  get id(): ClinicId { return this.clinic.id; }
  get name(): ClinicName { return this.clinic.name; }
  get licenseNumber(): LicenseNumber { return this.clinic.licenseNumber; }
  get address(): Address { return this.clinic.address; }
  get phone(): PhoneNumber { return this.clinic.phone; }
  get email(): EmailAddress { return this.clinic.email; }
  get website(): URL { return this.clinic.website; }
  get operatingHours(): readonly OperatingHours[] { return this.clinic.operatingHours; }
  get socialMedia(): readonly SocialMedia[] { return this.clinic.socialMedia; }
  get certifications(): readonly Certification[] { return this.clinic.certifications; }
  get services(): readonly MedicalProcedure[] { return this.clinic.services; }
  get practitioners(): readonly Practitioner[] { return this.clinic.practitioners; }
  get patientOutcomes(): readonly PatientOutcome[] { return this.clinic.patientOutcomes; }
  get overallRating(): Rating { return this.clinic.overallRating; }
  get totalReviews(): number { return this.clinic.totalReviews; }
  get establishedDate(): Date { return this.clinic.establishedDate; }
  get lastUpdated(): Date { return this.clinic.lastUpdated; }
  get extractionMetadata() { return this.clinic.extractionMetadata; }
}

// Utility functions
export const isClinicValid = (clinic: MedicalClinic): boolean => {
  return MedicalClinicSchema.safeParse(clinic).success;
};

export const compareClinics = (a: MedicalClinic, b: MedicalClinic): number => {
  // Compare by rating first, then by number of services
  const ratingComparison = b.overallRating.value - a.overallRating.value;
  if (ratingComparison !== 0) return ratingComparison;
  
  return b.services.length - a.services.length;
};

export const filterClinicsByLocation = (
  clinics: MedicalClinic[],
  city: string,
  radiusKm: number = 50
): MedicalClinic[] => {
  return clinics.filter(clinic => {
    if (clinic.address.city.toLowerCase() !== city.toLowerCase()) {
      return false;
    }
    
    // If coordinates are available, check radius
    if (clinic.address.coordinates) {
      // This would require a proper distance calculation
      // For now, just return true if city matches
      return true;
    }
    
    return true;
  });
};
