package com.cbruegg

import java.util.*

/**
 * Return a [Sequence] containing random values from
 * the [range].
 */
fun Random.intSequence(range: IntRange) =
        generateSequence { range.start + nextInt(range.endInclusive - range.start) }

/**
 * Generate randomized sublists.
 *
 * @param [howMany] Number of sublists to return in the [Sequence]
 * @param [ofSize] Size of the sublists.
 */
fun <T> List<T>.randomSubLists(howMany: Int, ofSize: Int, random: Random): Sequence<List<T>> {
    return if (ofSize == size) {
        return sequenceOf(this)
    } else generateSequence {
        random.intSequence(indices).distinct().take(ofSize).map { this[it] }.toList()
    }.distinct().take(howMany)
}