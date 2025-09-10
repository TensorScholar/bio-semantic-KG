/**
 * Branded Types - Type Safety for Domain Primitives
 * 
 * Provides branded type utilities for creating type-safe domain primitives
 * that prevent mixing different types of strings/numbers at compile time.
 * 
 * @file branded.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

/**
 * Branded type utility for creating type-safe primitives
 * 
 * @template T - The underlying type (string, number, etc.)
 * @template B - The brand identifier (unique string literal)
 */
export type Branded<T, B extends string> = T & { readonly __brand: B };

/**
 * Creates a branded type from a value
 * 
 * @param value - The value to brand
 * @returns The branded value
 */
export function brand<T, B extends string>(value: T): Branded<T, B> {
  return value as Branded<T, B>;
}

/**
 * Unwraps a branded type to its underlying value
 * 
 * @param branded - The branded value
 * @returns The underlying value
 */
export function unbrand<T, B extends string>(branded: Branded<T, B>): T {
  return branded as T;
}

/**
 * Type guard to check if a value is branded with a specific brand
 * 
 * @param value - The value to check
 * @param brand - The brand to check for
 * @returns True if the value has the specified brand
 */
export function isBranded<T, B extends string>(
  value: any,
  brand: B
): value is Branded<T, B> {
  return typeof value === 'object' && value !== null && '__brand' in value;
}

/**
 * Utility type for extracting the underlying type from a branded type
 */
export type Unbrand<T> = T extends Branded<infer U, any> ? U : T;

/**
 * Utility type for extracting the brand from a branded type
 */
export type Brand<T> = T extends Branded<any, infer B> ? B : never;

/**
 * Utility type for creating a union of branded types with the same underlying type
 */
export type BrandedUnion<T, B extends string> = Branded<T, B>;

/**
 * Utility type for creating an intersection of branded types
 */
export type BrandedIntersection<T1, B1 extends string, T2, B2 extends string> = 
  Branded<T1 & T2, `${B1}_${B2}`>;

/**
 * Factory function for creating branded type constructors
 * 
 * @param validator - Function to validate the underlying value
 * @returns A function that creates branded types
 */
export function createBrandedType<T, B extends string>(
  validator: (value: T) => boolean
) {
  return (value: T): Branded<T, B> => {
    if (!validator(value)) {
      throw new Error(`Invalid value for branded type: ${value}`);
    }
    return brand<T, B>(value);
  };
}

/**
 * Utility for creating branded string types with validation
 */
export function createBrandedString<B extends string>(
  validator: (value: string) => boolean
) {
  return createBrandedType<string, B>(validator);
}

/**
 * Utility for creating branded number types with validation
 */
export function createBrandedNumber<B extends string>(
  validator: (value: number) => boolean
) {
  return createBrandedType<number, B>(validator);
}

/**
 * Example usage and common branded types
 */

// Email address branded type
export type EmailAddress = Branded<string, "EmailAddress">;
export const createEmailAddress = createBrandedString<"EmailAddress">(
  (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
);

// Phone number branded type
export type PhoneNumber = Branded<string, "PhoneNumber">;
export const createPhoneNumber = createBrandedString<"PhoneNumber">(
  (value) => /^\+?[\d\s-()]+$/.test(value)
);

// URL branded type
export type URL = Branded<string, "URL">;
export const createURL = createBrandedString<"URL">(
  (value) => {
    try {
      new URL(value);
      return true;
    } catch {
      return false;
    }
  }
);

// Positive number branded type
export type PositiveNumber = Branded<number, "PositiveNumber">;
export const createPositiveNumber = createBrandedNumber<"PositiveNumber">(
  (value) => value > 0
);

// Non-negative number branded type
export type NonNegativeNumber = Branded<number, "NonNegativeNumber">;
export const createNonNegativeNumber = createBrandedNumber<"NonNegativeNumber">(
  (value) => value >= 0
);

// Integer branded type
export type Integer = Branded<number, "Integer">;
export const createInteger = createBrandedNumber<"Integer">(
  (value) => Number.isInteger(value)
);

// Positive integer branded type
export type PositiveInteger = Branded<number, "PositiveInteger">;
export const createPositiveInteger = createBrandedNumber<"PositiveInteger">(
  (value) => Number.isInteger(value) && value > 0
);

// Non-empty string branded type
export type NonEmptyString = Branded<string, "NonEmptyString">;
export const createNonEmptyString = createBrandedString<"NonEmptyString">(
  (value) => value.length > 0
);

// UUID branded type
export type UUID = Branded<string, "UUID">;
export const createUUID = createBrandedString<"UUID">(
  (value) => /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
);

// Date string branded type (ISO format)
export type ISODateString = Branded<string, "ISODateString">;
export const createISODateString = createBrandedString<"ISODateString">(
  (value) => {
    const date = new Date(value);
    return !isNaN(date.getTime()) && date.toISOString() === value;
  }
);

/**
 * Utility functions for working with branded types
 */

/**
 * Maps a branded type to another branded type
 */
export function mapBranded<T1, B1 extends string, T2, B2 extends string>(
  branded: Branded<T1, B1>,
  mapper: (value: T1) => T2
): Branded<T2, B2> {
  return brand<T2, B2>(mapper(unbrand(branded)));
}

/**
 * Chains branded type transformations
 */
export function chainBranded<T1, B1 extends string, T2, B2 extends string>(
  branded: Branded<T1, B1>,
  transformer: (value: T1) => Branded<T2, B2>
): Branded<T2, B2> {
  return transformer(unbrand(branded));
}

/**
 * Combines two branded types into a tuple
 */
export function combineBranded<T1, B1 extends string, T2, B2 extends string>(
  branded1: Branded<T1, B1>,
  branded2: Branded<T2, B2>
): Branded<[T1, T2], `${B1}_${B2}`> {
  return brand<[T1, T2], `${B1}_${B2}`>([unbrand(branded1), unbrand(branded2)]);
}

/**
 * Type-safe equality check for branded types
 */
export function brandedEquals<T, B extends string>(
  a: Branded<T, B>,
  b: Branded<T, B>
): boolean {
  return unbrand(a) === unbrand(b);
}

/**
 * Type-safe comparison for branded numbers
 */
export function brandedCompare<T extends number, B extends string>(
  a: Branded<T, B>,
  b: Branded<T, B>
): number {
  return unbrand(a) - unbrand(b);
}

/**
 * Utility for creating branded type arrays
 */
export function createBrandedArray<T, B extends string>(
  values: T[],
  brander: (value: T) => Branded<T, B>
): Branded<T, B>[] {
  return values.map(brander);
}

/**
 * Utility for filtering branded type arrays
 */
export function filterBranded<T, B extends string>(
  brandedArray: Branded<T, B>[],
  predicate: (value: T) => boolean
): Branded<T, B>[] {
  return brandedArray.filter(branded => predicate(unbrand(branded)));
}

/**
 * Utility for mapping branded type arrays
 */
export function mapBrandedArray<T1, B1 extends string, T2, B2 extends string>(
  brandedArray: Branded<T1, B1>[],
  mapper: (value: T1) => Branded<T2, B2>
): Branded<T2, B2>[] {
  return brandedArray.map(branded => mapper(unbrand(branded)));
}

/**
 * Utility for reducing branded type arrays
 */
export function reduceBranded<T, B extends string, R>(
  brandedArray: Branded<T, B>[],
  reducer: (accumulator: R, value: T) => R,
  initialValue: R
): R {
  return brandedArray.reduce((acc, branded) => reducer(acc, unbrand(branded)), initialValue);
}
