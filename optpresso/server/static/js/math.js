export function median(numbers) {
    const sorted = numbers.slice().sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return (sorted[middle - 1] + sorted[middle]) / 2;
    }

    return sorted[middle];
}


export function mean(numbers) {
    return numbers.reduce((a, b) => a + b, 0) / numbers.length;
}

export function std(numbers) {
    let avg = mean(numbers);
    return Math.sqrt(numbers.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / numbers.length);
}

