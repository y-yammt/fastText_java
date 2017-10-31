package cc.fasttext;

import java.util.Objects;

public final class Pair<K, V> {

    private K key_;
    private V value_;

    public Pair(K key, V value) {
        this.key_ = Objects.requireNonNull(key, "Null key");
        this.value_ = Objects.requireNonNull(value, "Null value");
    }

    public K first() {
        return key_;
    }

    public V second() {
        return value_;
    }

}
