/**
 * Option Type - Null Safety
 * 
 * Implements the Option pattern for null-safe programming without null/undefined.
 * Provides type-safe optional values with composable operations.
 * 
 * @file option.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

/**
 * Option type representing either Some value or None
 */
export type Option<T> = Some<T> | None<T>;

/**
 * Some variant of Option containing a value
 */
export class Some<T> {
  readonly _tag = "Some" as const;
  readonly _value: T;

  constructor(value: T) {
    this._value = value;
  }

  /**
   * Returns true if this is a Some variant
   */
  isSome(): this is Some<T> {
    return true;
  }

  /**
   * Returns false if this is a Some variant
   */
  isNone(): this is None<T> {
    return false;
  }

  /**
   * Returns the contained value
   */
  unwrap(): T {
    return this._value;
  }

  /**
   * Returns the contained value or the provided default
   */
  unwrapOr(defaultValue: T): T {
    return this._value;
  }

  /**
   * Returns the contained value or computes it from a function
   */
  unwrapOrElse(fn: () => T): T {
    return this._value;
  }

  /**
   * Maps the contained value using the provided function
   */
  map<U>(fn: (value: T) => U): Option<U> {
    return new Some(fn(this._value));
  }

  /**
   * Flat maps the contained value using the provided function
   */
  flatMap<U>(fn: (value: T) => Option<U>): Option<U> {
    return fn(this._value);
  }

  /**
   * Filters the contained value using the provided predicate
   */
  filter(predicate: (value: T) => boolean): Option<T> {
    return predicate(this._value) ? this : new None();
  }

  /**
   * Returns this Option if Some, otherwise returns the provided Option
   */
  or(other: Option<T>): Option<T> {
    return this;
  }

  /**
   * Returns this Option if Some, otherwise computes and returns another Option
   */
  orElse(fn: () => Option<T>): Option<T> {
    return this;
  }

  /**
   * Returns this Option if Some, otherwise returns the provided Option
   */
  and(other: Option<T>): Option<T> {
    return other;
  }

  /**
   * Returns this Option if Some, otherwise computes and returns another Option
   */
  andThen<U>(fn: (value: T) => Option<U>): Option<U> {
    return fn(this._value);
  }

  /**
   * Converts to Result, mapping None to the provided error
   */
  toResult<E>(error: E): Result<T, E> {
    return new Ok(this._value);
  }

  /**
   * Converts to Either, mapping None to the provided error
   */
  toEither<E>(error: E): Either<T, E> {
    return new Right(this._value);
  }

  /**
   * Matches on the Option variant
   */
  match<U>(matchers: {
    some: (value: T) => U;
    none: () => U;
  }): U {
    return matchers.some(this._value);
  }

  /**
   * Executes a function if this is Some
   */
  ifSome(fn: (value: T) => void): void {
    fn(this._value);
  }

  /**
   * Executes a function if this is None
   */
  ifNone(fn: () => void): void {
    // Do nothing for Some variant
  }

  /**
   * Returns a string representation
   */
  toString(): string {
    return `Some(${this._value})`;
  }

  /**
   * Returns JSON representation
   */
  toJSON(): { _tag: "Some"; _value: T } {
    return { _tag: "Some", _value: this._value };
  }
}

/**
 * None variant of Option
 */
export class None<T> {
  readonly _tag = "None" as const;

  /**
   * Returns false if this is a None variant
   */
  isSome(): this is Some<T> {
    return false;
  }

  /**
   * Returns true if this is a None variant
   */
  isNone(): this is None<T> {
    return true;
  }

  /**
   * Throws an error
   */
  unwrap(): never {
    throw new Error("Called unwrap on None");
  }

  /**
   * Returns the provided default value
   */
  unwrapOr(defaultValue: T): T {
    return defaultValue;
  }

  /**
   * Computes and returns a value from a function
   */
  unwrapOrElse(fn: () => T): T {
    return fn();
  }

  /**
   * Returns this None unchanged
   */
  map<U>(fn: (value: T) => U): Option<U> {
    return new None();
  }

  /**
   * Returns this None unchanged
   */
  flatMap<U>(fn: (value: T) => Option<U>): Option<U> {
    return new None();
  }

  /**
   * Returns this None unchanged
   */
  filter(predicate: (value: T) => boolean): Option<T> {
    return this;
  }

  /**
   * Returns the provided Option
   */
  or(other: Option<T>): Option<T> {
    return other;
  }

  /**
   * Computes and returns another Option
   */
  orElse(fn: () => Option<T>): Option<T> {
    return fn();
  }

  /**
   * Returns this None unchanged
   */
  and(other: Option<T>): Option<T> {
    return this;
  }

  /**
   * Returns this None unchanged
   */
  andThen<U>(fn: (value: T) => Option<U>): Option<U> {
    return new None();
  }

  /**
   * Converts to Result, mapping None to the provided error
   */
  toResult<E>(error: E): Result<T, E> {
    return new Err(error);
  }

  /**
   * Converts to Either, mapping None to the provided error
   */
  toEither<E>(error: E): Either<T, E> {
    return new Left(error);
  }

  /**
   * Matches on the Option variant
   */
  match<U>(matchers: {
    some: (value: T) => U;
    none: () => U;
  }): U {
    return matchers.none();
  }

  /**
   * Executes a function if this is Some
   */
  ifSome(fn: (value: T) => void): void {
    // Do nothing for None variant
  }

  /**
   * Executes a function if this is None
   */
  ifNone(fn: () => void): void {
    fn();
  }

  /**
   * Returns a string representation
   */
  toString(): string {
    return "None";
  }

  /**
   * Returns JSON representation
   */
  toJSON(): { _tag: "None" } {
    return { _tag: "None" };
  }
}

/**
 * Creates a Some Option
 */
export function Some<T>(value: T): Some<T> {
  return new Some(value);
}

/**
 * Creates a None Option
 */
export function None<T>(): None<T> {
  return new None();
}

/**
 * Creates an Option from a nullable value
 */
export function fromNullable<T>(value: T | null | undefined): Option<T> {
  return value == null ? new None() : new Some(value);
}

/**
 * Creates an Option from a value that might be null/undefined
 */
export function fromValue<T>(value: T | null | undefined): Option<T> {
  return value == null ? new None() : new Some(value);
}

/**
 * Creates an Option from a predicate
 */
export function fromPredicate<T>(
  value: T,
  predicate: (value: T) => boolean
): Option<T> {
  return predicate(value) ? new Some(value) : new None();
}

/**
 * Creates an Option from a function that might throw
 */
export function tryCatch<T>(fn: () => T): Option<T> {
  try {
    return new Some(fn());
  } catch {
    return new None();
  }
}

/**
 * Creates an Option from an async function that might throw
 */
export async function tryCatchAsync<T>(fn: () => Promise<T>): Promise<Option<T>> {
  try {
    const value = await fn();
    return new Some(value);
  } catch {
    return new None();
  }
}

/**
 * Combines multiple Options into a single Option
 */
export function combine<T extends readonly unknown[]>(
  options: { [K in keyof T]: Option<T[K]> }
): Option<T> {
  const values: T = [] as unknown as T;
  
  for (let i = 0; i < options.length; i++) {
    const option = options[i];
    if (option.isNone()) {
      return new None();
    }
    values[i] = option._value;
  }
  
  return new Some(values);
}

/**
 * Combines multiple Options into a single Option, collecting all values
 */
export function combineAll<T>(options: Option<T>[]): Option<T[]> {
  const values: T[] = [];
  
  for (const option of options) {
    if (option.isNone()) {
      return new None();
    }
    values.push(option._value);
  }
  
  return new Some(values);
}

/**
 * Filters Options, keeping only Some values
 */
export function filterSome<T>(options: Option<T>[]): T[] {
  return options
    .filter((option): option is Some<T> => option.isSome())
    .map(option => option._value);
}

/**
 * Filters Options, keeping only None values
 */
export function filterNone<T>(options: Option<T>[]): None<T>[] {
  return options.filter((option): option is None<T> => option.isNone());
}

/**
 * Partitions Options into Some and None arrays
 */
export function partition<T>(options: Option<T>[]): [T[], None<T>[]] {
  const someValues: T[] = [];
  const noneValues: None<T>[] = [];
  
  for (const option of options) {
    if (option.isSome()) {
      someValues.push(option._value);
    } else {
      noneValues.push(option);
    }
  }
  
  return [someValues, noneValues];
}

/**
 * Maps over an array of Options
 */
export function mapOptions<T, U>(
  options: Option<T>[],
  fn: (value: T) => U
): Option<U[]> {
  const mapped: U[] = [];
  
  for (const option of options) {
    if (option.isNone()) {
      return new None();
    }
    mapped.push(fn(option._value));
  }
  
  return new Some(mapped);
}

/**
 * Flat maps over an array of Options
 */
export function flatMapOptions<T, U>(
  options: Option<T>[],
  fn: (value: T) => Option<U>
): Option<U[]> {
  const mapped: U[] = [];
  
  for (const option of options) {
    if (option.isNone()) {
      return new None();
    }
    const mappedOption = fn(option._value);
    if (mappedOption.isNone()) {
      return new None();
    }
    mapped.push(mappedOption._value);
  }
  
  return new Some(mapped);
}

/**
 * Reduces an array of Options
 */
export function reduceOptions<T, U>(
  options: Option<T>[],
  initialValue: U,
  fn: (accumulator: U, value: T) => U
): Option<U> {
  let accumulator = initialValue;
  
  for (const option of options) {
    if (option.isNone()) {
      return new None();
    }
    accumulator = fn(accumulator, option._value);
  }
  
  return new Some(accumulator);
}

/**
 * Finds the first Some value in an array of Options
 */
export function findSome<T>(options: Option<T>[]): Option<T> {
  for (const option of options) {
    if (option.isSome()) {
      return option;
    }
  }
  return new None();
}

/**
 * Finds the first Some value matching a predicate
 */
export function findSomeWhere<T>(
  options: Option<T>[],
  predicate: (value: T) => boolean
): Option<T> {
  for (const option of options) {
    if (option.isSome() && predicate(option._value)) {
      return option;
    }
  }
  return new None();
}

/**
 * Type guard to check if a value is an Option
 */
export function isOption<T>(value: unknown): value is Option<T> {
  return (
    typeof value === "object" &&
    value !== null &&
    ("_tag" in value) &&
    (value._tag === "Some" || value._tag === "None")
  );
}

/**
 * Type guard to check if a value is a Some Option
 */
export function isSome<T>(value: unknown): value is Some<T> {
  return isOption(value) && value._tag === "Some";
}

/**
 * Type guard to check if a value is a None Option
 */
export function isNone<T>(value: unknown): value is None<T> {
  return isOption(value) && value._tag === "None";
}

// Import Result and Either for type compatibility
import type { Result, Ok, Err } from "./result.ts";
import type { Either, Left, Right } from "./either.ts";
