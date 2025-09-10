/**
 * Result Type - Functional Error Handling
 * 
 * Implements the Result pattern for functional error handling without exceptions.
 * Provides type-safe error handling with composable operations.
 * 
 * @file result.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

/**
 * Result type representing either success (Ok) or failure (Err)
 */
export type Result<T, E = Error> = Ok<T, E> | Err<T, E>;

/**
 * Success variant of Result
 */
export class Ok<T, E = Error> {
  readonly _tag = "Ok" as const;
  readonly _value: T;

  constructor(value: T) {
    this._value = value;
  }

  /**
   * Returns true if this is an Ok variant
   */
  isOk(): this is Ok<T, E> {
    return true;
  }

  /**
   * Returns false if this is an Ok variant
   */
  isErr(): this is Err<T, E> {
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
  unwrapOrElse(fn: (error: E) => T): T {
    return this._value;
  }

  /**
   * Maps the contained value using the provided function
   */
  map<U>(fn: (value: T) => U): Result<U, E> {
    return new Ok(fn(this._value));
  }

  /**
   * Maps the contained error using the provided function
   */
  mapErr<F>(fn: (error: E) => F): Result<T, F> {
    return new Ok(this._value);
  }

  /**
   * Chains another Result operation
   */
  andThen<U, F>(fn: (value: T) => Result<U, F>): Result<U, E | F> {
    return fn(this._value);
  }

  /**
   * Returns this Result if Ok, otherwise returns the provided Result
   */
  or<U>(other: Result<U, E>): Result<T | U, E> {
    return this;
  }

  /**
   * Returns this Result if Ok, otherwise computes and returns another Result
   */
  orElse<U>(fn: (error: E) => Result<U, E>): Result<T | U, E> {
    return this;
  }

  /**
   * Converts to Option, discarding error information
   */
  toOption(): Option<T> {
    return new Some(this._value);
  }

  /**
   * Converts to Either
   */
  toEither(): Either<T, E> {
    return new Right(this._value);
  }

  /**
   * Matches on the Result variant
   */
  match<U>(matchers: {
    ok: (value: T) => U;
    err: (error: E) => U;
  }): U {
    return matchers.ok(this._value);
  }

  /**
   * Executes a function if this is Ok
   */
  ifOk(fn: (value: T) => void): void {
    fn(this._value);
  }

  /**
   * Executes a function if this is Err
   */
  ifErr(fn: (error: E) => void): void {
    // Do nothing for Ok variant
  }

  /**
   * Returns a string representation
   */
  toString(): string {
    return `Ok(${this._value})`;
  }

  /**
   * Returns JSON representation
   */
  toJSON(): { _tag: "Ok"; _value: T } {
    return { _tag: "Ok", _value: this._value };
  }
}

/**
 * Error variant of Result
 */
export class Err<T, E = Error> {
  readonly _tag = "Err" as const;
  readonly _error: E;

  constructor(error: E) {
    this._error = error;
  }

  /**
   * Returns false if this is an Err variant
   */
  isOk(): this is Ok<T, E> {
    return false;
  }

  /**
   * Returns true if this is an Err variant
   */
  isErr(): this is Err<T, E> {
    return true;
  }

  /**
   * Throws the contained error
   */
  unwrap(): never {
    throw this._error;
  }

  /**
   * Returns the provided default value
   */
  unwrapOr(defaultValue: T): T {
    return defaultValue;
  }

  /**
   * Computes and returns a value from the error
   */
  unwrapOrElse(fn: (error: E) => T): T {
    return fn(this._error);
  }

  /**
   * Returns this Err unchanged
   */
  map<U>(fn: (value: T) => U): Result<U, E> {
    return new Err(this._error);
  }

  /**
   * Maps the contained error using the provided function
   */
  mapErr<F>(fn: (error: E) => F): Result<T, F> {
    return new Err(fn(this._error));
  }

  /**
   * Returns this Err unchanged
   */
  andThen<U, F>(fn: (value: T) => Result<U, F>): Result<U, E | F> {
    return new Err(this._error);
  }

  /**
   * Returns the provided Result
   */
  or<U>(other: Result<U, E>): Result<T | U, E> {
    return other;
  }

  /**
   * Computes and returns another Result
   */
  orElse<U>(fn: (error: E) => Result<U, E>): Result<T | U, E> {
    return fn(this._error);
  }

  /**
   * Converts to None
   */
  toOption(): Option<T> {
    return new None();
  }

  /**
   * Converts to Either
   */
  toEither(): Either<T, E> {
    return new Left(this._error);
  }

  /**
   * Matches on the Result variant
   */
  match<U>(matchers: {
    ok: (value: T) => U;
    err: (error: E) => U;
  }): U {
    return matchers.err(this._error);
  }

  /**
   * Executes a function if this is Ok
   */
  ifOk(fn: (value: T) => void): void {
    // Do nothing for Err variant
  }

  /**
   * Executes a function if this is Err
   */
  ifErr(fn: (error: E) => void): void {
    fn(this._error);
  }

  /**
   * Returns a string representation
   */
  toString(): string {
    return `Err(${this._error})`;
  }

  /**
   * Returns JSON representation
   */
  toJSON(): { _tag: "Err"; _error: E } {
    return { _tag: "Err", _error: this._error };
  }
}

/**
 * Creates an Ok Result
 */
export function Ok<T, E = Error>(value: T): Ok<T, E> {
  return new Ok(value);
}

/**
 * Creates an Err Result
 */
export function Err<T, E = Error>(error: E): Err<T, E> {
  return new Err(error);
}

/**
 * Wraps a function that might throw in a Result
 */
export function tryCatch<T, E = Error>(
  fn: () => T,
  errorMapper?: (error: unknown) => E
): Result<T, E> {
  try {
    return new Ok(fn());
  } catch (error) {
    const mappedError = errorMapper ? errorMapper(error) : (error as E);
    return new Err(mappedError);
  }
}

/**
 * Wraps an async function that might throw in a Result
 */
export async function tryCatchAsync<T, E = Error>(
  fn: () => Promise<T>,
  errorMapper?: (error: unknown) => E
): Promise<Result<T, E>> {
  try {
    const value = await fn();
    return new Ok(value);
  } catch (error) {
    const mappedError = errorMapper ? errorMapper(error) : (error as E);
    return new Err(mappedError);
  }
}

/**
 * Combines multiple Results into a single Result
 */
export function combine<T extends readonly unknown[], E>(
  results: { [K in keyof T]: Result<T[K], E> }
): Result<T, E> {
  const values: T = [] as unknown as T;
  
  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    if (result.isErr()) {
      return new Err(result._error);
    }
    values[i] = result._value;
  }
  
  return new Ok(values);
}

/**
 * Combines multiple Results into a single Result, collecting all errors
 */
export function combineAll<T extends readonly unknown[], E>(
  results: { [K in keyof T]: Result<T[K], E> }
): Result<T, E[]> {
  const values: T = [] as unknown as T;
  const errors: E[] = [];
  
  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    if (result.isErr()) {
      errors.push(result._error);
    } else {
      values[i] = result._value;
    }
  }
  
  if (errors.length > 0) {
    return new Err(errors);
  }
  
  return new Ok(values);
}

/**
 * Filters Results, keeping only Ok values
 */
export function filterOk<T, E>(results: Result<T, E>[]): T[] {
  return results
    .filter((result): result is Ok<T, E> => result.isOk())
    .map(result => result._value);
}

/**
 * Filters Results, keeping only Err values
 */
export function filterErr<T, E>(results: Result<T, E>[]): E[] {
  return results
    .filter((result): result is Err<T, E> => result.isErr())
    .map(result => result._error);
}

/**
 * Partitions Results into Ok and Err arrays
 */
export function partition<T, E>(results: Result<T, E>[]): [T[], E[]] {
  const okValues: T[] = [];
  const errValues: E[] = [];
  
  for (const result of results) {
    if (result.isOk()) {
      okValues.push(result._value);
    } else {
      errValues.push(result._error);
    }
  }
  
  return [okValues, errValues];
}

/**
 * Maps over an array of Results
 */
export function mapResults<T, U, E>(
  results: Result<T, E>[],
  fn: (value: T) => U
): Result<U[], E> {
  const mapped: U[] = [];
  
  for (const result of results) {
    if (result.isErr()) {
      return new Err(result._error);
    }
    mapped.push(fn(result._value));
  }
  
  return new Ok(mapped);
}

/**
 * Flat maps over an array of Results
 */
export function flatMapResults<T, U, E>(
  results: Result<T, E>[],
  fn: (value: T) => Result<U, E>
): Result<U[], E> {
  const mapped: U[] = [];
  
  for (const result of results) {
    if (result.isErr()) {
      return new Err(result._error);
    }
    const mappedResult = fn(result._value);
    if (mappedResult.isErr()) {
      return new Err(mappedResult._error);
    }
    mapped.push(mappedResult._value);
  }
  
  return new Ok(mapped);
}

/**
 * Reduces an array of Results
 */
export function reduceResults<T, U, E>(
  results: Result<T, E>[],
  initialValue: U,
  fn: (accumulator: U, value: T) => U
): Result<U, E> {
  let accumulator = initialValue;
  
  for (const result of results) {
    if (result.isErr()) {
      return new Err(result._error);
    }
    accumulator = fn(accumulator, result._value);
  }
  
  return new Ok(accumulator);
}

/**
 * Type guard to check if a value is a Result
 */
export function isResult<T, E>(value: unknown): value is Result<T, E> {
  return (
    typeof value === "object" &&
    value !== null &&
    ("_tag" in value) &&
    (value._tag === "Ok" || value._tag === "Err")
  );
}

/**
 * Type guard to check if a value is an Ok Result
 */
export function isOk<T, E>(value: unknown): value is Ok<T, E> {
  return isResult(value) && value._tag === "Ok";
}

/**
 * Type guard to check if a value is an Err Result
 */
export function isErr<T, E>(value: unknown): value is Err<T, E> {
  return isResult(value) && value._tag === "Err";
}

// Import Option and Either for type compatibility
import type { Option, Some, None } from "./option.ts";
import type { Either, Left, Right } from "./either.ts";
