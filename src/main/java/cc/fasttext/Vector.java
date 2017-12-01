package cc.fasttext;

import java.util.List;
import java.util.Objects;
import java.util.StringJoiner;

import org.apache.commons.lang.Validate;
import org.apache.commons.math3.util.FastMath;

import com.google.common.primitives.Floats;

public strictfp class Vector {

    public int m_;
    public float[] data_;

    public Vector(int size) {
        m_ = size;
        data_ = new float[size];
    }

    public int size() {
        return m_;
    }

    public void zero() {
        data_ = new float[m_];
    }

    /**
     * <pre>{@code real Vector::norm() const {
     *  real sum = 0;
     *  for (int64_t i = 0; i < m_; i++) {
     *      sum += data_[i] * data_[i];
     *  }
     *  return std::sqrt(sum);
     * }
     * }</pre>
     *
     * @return float
     */
    public float norm() {
        double sum = 0;
        for (int i = 0; i < m_; i++) {
            sum += data_[i] * data_[i];
        }
        return (float) FastMath.sqrt(sum);
    }

    /**
     * <pre>{@code void Vector::addVector(const Vector& source) {
     *  assert(m_ == source.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] += source.data_[i];
     *  }
     * }}</pre>
     *
     * @param source {@link Vector}
     */
    public void addVector(Vector source) {
        Validate.isTrue(m_ == Objects.requireNonNull(source, "Null source vector").m_, "Wrong size of vector: " + m_ + "!=" + source.m_);
        for (int i = 0; i < m_; i++) {
            data_[i] += source.data_[i];
        }
    }

    /**
     * <pre>{@code void Vector::addVector(const Vector& source, real s) {
     *  assert(m_ == source.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] += s * source.data_[i];
     *  }
     * }}</pre>
     *
     * @param source
     * @param s
     */
    public void addVector(Vector source, float s) {
        Validate.isTrue(m_ == Objects.requireNonNull(source, "Null source vector").m_, "Wrong size of vector: " + m_ + "!=" + source.m_);
        for (int i = 0; i < m_; i++) {
            data_[i] += s * source.data_[i];
        }
    }

    /**
     * <pre>{@code
     * void Vector::mul(real a) {
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] *= a;
     *  }
     * }}</pre>
     *
     * @param a
     */
    public void mul(float a) {
        for (int i = 0; i < m_; i++) {
            data_[i] *= a;
        }
    }

    /**
     * <pre>{@code
     * void Vector::addRow(const Matrix& A, int64_t i) {
     *  assert(i >= 0);
     *  assert(i < A.m_);
     *  assert(m_ == A.n_);
     *  for (int64_t j = 0; j < A.n_; j++) {
     *      data_[j] += A.at(i, j);
     *  }
     * }}</pre>
     *
     * @param matrix
     * @param i
     */
    public void addRow(Matrix matrix, int i) {
        Validate.isTrue(i >= 0 && i < matrix.getM(), "Incompatible index (" + i + ") and matrix m-size (" + matrix.getM() + ")");
        Validate.isTrue(m_ == matrix.getN(), "Wrong matrix n-size: " + m_ + " != " + matrix.getN());
        for (int j = 0; j < matrix.getN(); j++) {
            data_[j] += matrix.at(i, j);
        }
    }

    /**
     * <pre>{@code
     * void Vector::addRow(const Matrix& A, int64_t i, real a) {
     *  assert(i >= 0);
     *  assert(i < A.m_);
     *  assert(m_ == A.n_);
     *  for (int64_t j = 0; j < A.n_; j++) {
     *      data_[j] += a * A.at(i, j);
     *  }
     * }
     * }</pre>
     *
     * @param matrix
     * @param i
     * @param a
     */
    public void addRow(Matrix matrix, int i, float a) {
        Validate.isTrue(i >= 0 && i < matrix.getM(), "Incompatible index (" + i + ") and matrix m-size (" + matrix.getM() + ")");
        Validate.isTrue(m_ == matrix.getN(), "Wrong matrix n-size: " + m_ + " != " + matrix.getN());
        for (int j = 0; j < matrix.getN(); j++) {
            data_[j] += a * matrix.at(i, j);
        }
    }

    /**
     * <pre>{@code
     * void Vector::mul(const Matrix& A, const Vector& vec) {
     *  assert(A.m_ == m_);
     *  assert(A.n_ == vec.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] = A.dotRow(vec, i);
     *  }
     * }}</pre>
     *
     * @param matrix
     * @param vector
     */
    public void mul(Matrix matrix, Vector vector) {
        Validate.isTrue(matrix.getM() == m_, "Wrong matrix m-size: " + m_ + " != " + matrix.getM());
        Validate.isTrue(matrix.getN() == vector.m_, "Matrix n-size (" + matrix.getN() + ") and vector size (" + vector.m_ + ")  are not equal.");
        for (int i = 0; i < m_; i++) {
            data_[i] = matrix.dotRow(vector, i);
        }
    }

    public int argmax() {
        float max = data_[0];
        int argmax = 0;
        for (int i = 1; i < m_; i++) {
            if (data_[i] > max) {
                max = data_[i];
                argmax = i;
            }
        }
        return argmax;
    }

    public float get(int i) {
        return data_[i];
    }

    public void set(int i, float value) {
        data_[i] = value;
    }

    public List<Float> getData() {
        return Floats.asList(data_);
    }

    /**
     * todo: fix.
     * see {@link Utils#formatNumber(float)}
     *
     * @return
     */
    @Override
    public String toString() {
        StringJoiner res = new StringJoiner(" ");
        for (float data : data_) {
            res.add(Utils.formatNumber(data));
        }
        return res.toString();
    }

}
